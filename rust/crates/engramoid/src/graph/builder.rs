//! AST graph builder using tree-sitter for Python, Rust, TypeScript, Go,
//! and Java source files.
//!
//! Builds a `GraphEngine` with File, Module, Class, Function nodes and
//! Contains, Calls, Imports edges from tree-sitter AST parsing. Falls back
//! to skipping files on parse errors (graceful degradation).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::graph::engine::GraphEngine;
use crate::graph::models::{Edge, EdgeKind, MetaValue, Node, NodeId, NodeKind};
use crate::retrieval::chunks::Chunk;

/// Languages supported by the AST graph builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lang {
    Python,
    Rust,
    TypeScript,
    Go,
    Java,
}

impl Lang {
    fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "py" => Some(Lang::Python),
            "rs" => Some(Lang::Rust),
            "ts" | "tsx" => Some(Lang::TypeScript),
            "go" => Some(Lang::Go),
            "java" => Some(Lang::Java),
            _ => None,
        }
    }

    fn language(self) -> tree_sitter::Language {
        match self {
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Lang::Go => tree_sitter_go::LANGUAGE.into(),
            Lang::Java => tree_sitter_java::LANGUAGE.into(),
        }
    }

    /// Node kinds that can be embedded (function/class equivalents).
    #[allow(dead_code)]
    fn embeddable_kinds() -> &'static [&'static str] {
        &[
            // Python
            "function_definition", "class_definition",
            // Rust
            "function_item", "struct_item", "impl_item",
            // TypeScript
            "function_declaration", "method_definition", "class_declaration",
            // Go
            "function_declaration", "method_declaration", "type_declaration",
            // Java
            "method_declaration", "class_declaration", "interface_declaration",
        ]
    }
}

/// Builds a `GraphEngine` from source files using tree-sitter.
pub struct AstGraphBuilder {
    pub max_file_size_bytes: u64,
}

impl Default for AstGraphBuilder {
    fn default() -> Self {
        Self {
            max_file_size_bytes: 256 * 1024,
        }
    }
}

impl AstGraphBuilder {
    /// Walk `repo_root`, parse source files with tree-sitter, and populate
    /// a `GraphEngine` with File → Module → Function/Class containment,
    /// cross-file `Calls` edges, and import references.
    #[must_use]
    pub fn build(&self, repo_root: &Path) -> GraphEngine {
        let mut engine = GraphEngine::new();
        let mut func_name_to_ids: HashMap<String, Vec<NodeId>> = HashMap::new();

        for entry in WalkDir::new(repo_root)
            .max_depth(20)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| !is_skip_dir(n))
            })
            .flatten()
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            let Some(lang) = Lang::from_extension(ext) else {
                continue;
            };

            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.len() > self.max_file_size_bytes {
                continue;
            };

            let Ok(source) = std::fs::read_to_string(path) else {
                continue;
            };

            let rel_path = path
                .strip_prefix(repo_root)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();

            self.process_file(
                &rel_path,
                &source,
                lang,
                &mut engine,
                &mut func_name_to_ids,
            );
        }

        engine
    }

    /// Extract chunk texts for Function and Class nodes so they can be
    /// embedded. Re-reads source files from disk using byte-range metadata
    /// stored on each node during `build()`.
    #[must_use]
    pub fn extract_chunks(
        &self,
        engine: &GraphEngine,
        repo_root: &Path,
    ) -> Vec<(NodeId, Chunk)> {
        let mut out = Vec::new();
        let mut file_cache: HashMap<PathBuf, String> = HashMap::new();

        for node in engine.all_nodes() {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Class) {
                continue;
            }
            let Some(MetaValue::Text(file_path)) = node.get_meta("file") else {
                continue;
            };
            let Some(MetaValue::Number(start_byte)) = node.get_meta("start_byte") else {
                continue;
            };
            let Some(MetaValue::Number(end_byte)) = node.get_meta("end_byte") else {
                continue;
            };

            let full_path = repo_root.join(file_path.as_str());
            let source = if let Some(cached) = file_cache.get(&full_path) {
                cached
            } else {
                let Ok(content) = std::fs::read_to_string(&full_path) else {
                    continue;
                };
                file_cache.insert(full_path.clone(), content);
                file_cache.get(&full_path).unwrap()
            };

            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let start = (*start_byte) as usize;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let end = (*end_byte as usize).min(source.len());
            if start >= end || end > source.len() {
                continue;
            }

            let text = source[start..end].to_string();
            let line_start = source[..start].lines().count() + 1;
            let line_end = source[..end].lines().count();

            out.push((
                node.id,
                Chunk {
                    path: file_path.clone(),
                    line_start,
                    line_end,
                    text,
                },
            ));
        }
        out
    }

    // ── per-file processing ──────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn process_file(
        &self,
        rel_path: &str,
        source: &str,
        lang: Lang,
        engine: &mut GraphEngine,
        func_name_to_ids: &mut HashMap<String, Vec<NodeId>>,
    ) {
        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(&lang.language()).is_err() {
            return;
        }

        let Some(tree) = parser.parse(source, None) else {
            return;
        };

        let root = tree.root_node();

        let file_id = add_node(engine, NodeKind::File, rel_path, &format!("File: {rel_path}"));
        let module_id = add_node(
            engine,
            NodeKind::Module,
            &format!("module:{rel_path}"),
            &format!("Module in {rel_path}"),
        );
        engine.add_edge(Edge::new(file_id, module_id, EdgeKind::Contains));

        // Dispatch top-level children based on language
        match lang {
            Lang::Python => self.process_python(
                root, source, rel_path, module_id, file_id, engine, func_name_to_ids,
            ),
            Lang::Rust => self.process_rust(
                root, source, rel_path, module_id, file_id, engine, func_name_to_ids,
            ),
            Lang::TypeScript => self.process_typescript(
                root, source, rel_path, module_id, file_id, engine, func_name_to_ids,
            ),
            Lang::Go => self.process_go(
                root, source, rel_path, module_id, file_id, engine, func_name_to_ids,
            ),
            Lang::Java => self.process_java(
                root, source, rel_path, module_id, file_id, engine, func_name_to_ids,
            ),
        }
    }

    // ── Python ───────────────────────────────────────────────────────────

    fn process_python(
        &self,
        root: tree_sitter::Node,
        source: &str,
        file_path: &str,
        module_id: NodeId,
        file_id: NodeId,
        engine: &mut GraphEngine,
        func_name_to_ids: &mut HashMap<String, Vec<NodeId>>,
    ) {
        for i in 0..root.named_child_count() {
            let Some(child) = root.named_child(i) else { continue };
            match child.kind() {
                "function_definition" => {
                    let fid = process_py_function(child, source, file_path, module_id, engine);
                    func_name_to_ids
                        .entry(child_name(child, source))
                        .or_default()
                        .push(fid);
                }
                "class_definition" => {
                    let cid = process_py_class(child, source, file_path, module_id, engine, func_name_to_ids);
                    func_name_to_ids
                        .entry(child_name(child, source))
                        .or_default()
                        .push(cid);
                }
                "import_statement" | "import_from_statement" => {
                    process_py_import(child, source, file_id, engine);
                }
                _ => {}
            }
        }
    }

    // ── Rust ─────────────────────────────────────────────────────────────

    fn process_rust(
        &self,
        root: tree_sitter::Node,
        source: &str,
        file_path: &str,
        module_id: NodeId,
        _file_id: NodeId,
        engine: &mut GraphEngine,
        func_name_to_ids: &mut HashMap<String, Vec<NodeId>>,
    ) {
        for i in 0..root.named_child_count() {
            let Some(child) = root.named_child(i) else { continue };
            match child.kind() {
                "function_item" => {
                    let name = child_name(child, source);
                    let fid = add_node(engine, NodeKind::Function, &name, &format!("fn {name} in {file_path}"));
                    set_byte_meta(engine, &fid, file_path, child);
                    engine.add_edge(Edge::new(module_id, fid, EdgeKind::Contains));
                    func_name_to_ids.entry(name).or_default().push(fid);
                }
                "struct_item" => {
                    let name = child_name(child, source);
                    let sid = add_node(engine, NodeKind::Class, &name, &format!("struct {name} in {file_path}"));
                    set_byte_meta(engine, &sid, file_path, child);
                    engine.add_edge(Edge::new(module_id, sid, EdgeKind::Contains));
                    func_name_to_ids.entry(name).or_default().push(sid);
                }
                "impl_item" => {
                    let name = child_name(child, source);
                    let iid = add_node(engine, NodeKind::Class, &name, &format!("impl {name} in {file_path}"));
                    set_byte_meta(engine, &iid, file_path, child);
                    engine.add_edge(Edge::new(module_id, iid, EdgeKind::Contains));
                    // Process methods inside impl
                    if let Some(body) = child.child_by_field_name("body") {
                        for j in 0..body.named_child_count() {
                            if let Some(method) = body.named_child(j) {
                                if method.kind() == "function_item" {
                                    let mname = child_name(method, source);
                                    let mid = add_node(engine, NodeKind::Function, &mname, &format!("fn {mname} in {file_path}"));
                                    set_byte_meta(engine, &mid, file_path, method);
                                    engine.add_edge(Edge::new(iid, mid, EdgeKind::Contains));
                                    func_name_to_ids.entry(mname).or_default().push(mid);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // ── TypeScript ───────────────────────────────────────────────────────

    fn process_typescript(
        &self,
        root: tree_sitter::Node,
        source: &str,
        file_path: &str,
        module_id: NodeId,
        _file_id: NodeId,
        engine: &mut GraphEngine,
        func_name_to_ids: &mut HashMap<String, Vec<NodeId>>,
    ) {
        for i in 0..root.named_child_count() {
            let Some(child) = root.named_child(i) else { continue };
            match child.kind() {
                "function_declaration" | "method_definition" => {
                    let name = child_name(child, source);
                    let fid = add_node(engine, NodeKind::Function, &name, &format!("function {name} in {file_path}"));
                    set_byte_meta(engine, &fid, file_path, child);
                    engine.add_edge(Edge::new(module_id, fid, EdgeKind::Contains));
                    func_name_to_ids.entry(name).or_default().push(fid);
                }
                "class_declaration" => {
                    let name = child_name(child, source);
                    let cid = add_node(engine, NodeKind::Class, &name, &format!("class {name} in {file_path}"));
                    set_byte_meta(engine, &cid, file_path, child);
                    engine.add_edge(Edge::new(module_id, cid, EdgeKind::Contains));
                    func_name_to_ids.entry(name).or_default().push(cid);
                    // Process class body for methods
                    if let Some(body) = child.child_by_field_name("body") {
                        process_ts_class_body(body, source, file_path, cid, engine, func_name_to_ids);
                    }
                }
                _ => {}
            }
        }
    }

    // ── Go ───────────────────────────────────────────────────────────────

    fn process_go(
        &self,
        root: tree_sitter::Node,
        source: &str,
        file_path: &str,
        module_id: NodeId,
        _file_id: NodeId,
        engine: &mut GraphEngine,
        func_name_to_ids: &mut HashMap<String, Vec<NodeId>>,
    ) {
        for i in 0..root.named_child_count() {
            let Some(child) = root.named_child(i) else { continue };
            match child.kind() {
                "function_declaration" | "method_declaration" => {
                    let name = child_name(child, source);
                    let fid = add_node(engine, NodeKind::Function, &name, &format!("func {name} in {file_path}"));
                    set_byte_meta(engine, &fid, file_path, child);
                    engine.add_edge(Edge::new(module_id, fid, EdgeKind::Contains));
                    func_name_to_ids.entry(name).or_default().push(fid);
                }
                "type_declaration" => {
                    // Go type declarations can be structs or interfaces
                    let name = child_name(child, source);
                    let tid = add_node(engine, NodeKind::Class, &name, &format!("type {name} in {file_path}"));
                    set_byte_meta(engine, &tid, file_path, child);
                    engine.add_edge(Edge::new(module_id, tid, EdgeKind::Contains));
                    func_name_to_ids.entry(name).or_default().push(tid);
                }
                _ => {}
            }
        }
    }

    // ── Java ─────────────────────────────────────────────────────────────

    fn process_java(
        &self,
        root: tree_sitter::Node,
        source: &str,
        file_path: &str,
        module_id: NodeId,
        _file_id: NodeId,
        engine: &mut GraphEngine,
        func_name_to_ids: &mut HashMap<String, Vec<NodeId>>,
    ) {
        for i in 0..root.named_child_count() {
            let Some(child) = root.named_child(i) else { continue };
            match child.kind() {
                "method_declaration" => {
                    let name = child_name(child, source);
                    let mid = add_node(engine, NodeKind::Function, &name, &format!("method {name} in {file_path}"));
                    set_byte_meta(engine, &mid, file_path, child);
                    engine.add_edge(Edge::new(module_id, mid, EdgeKind::Contains));
                    func_name_to_ids.entry(name).or_default().push(mid);
                }
                "class_declaration" => {
                    let name = child_name(child, source);
                    let cid = add_node(engine, NodeKind::Class, &name, &format!("class {name} in {file_path}"));
                    set_byte_meta(engine, &cid, file_path, child);
                    engine.add_edge(Edge::new(module_id, cid, EdgeKind::Contains));
                    func_name_to_ids.entry(name).or_default().push(cid);
                    // Process class body for methods
                    if let Some(body) = child.child_by_field_name("body") {
                        for j in 0..body.named_child_count() {
                            if let Some(member) = body.named_child(j) {
                                if member.kind() == "method_declaration" {
                                    let mname = child_name(member, source);
                                    let mid = add_node(engine, NodeKind::Function, &mname, &format!("method {mname} in {file_path}"));
                                    set_byte_meta(engine, &mid, file_path, member);
                                    engine.add_edge(Edge::new(cid, mid, EdgeKind::Contains));
                                    func_name_to_ids.entry(mname).or_default().push(mid);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

// ── Python-specific helpers ──────────────────────────────────────────────

fn process_py_function(
    node: tree_sitter::Node,
    source: &str,
    file_path: &str,
    parent_id: NodeId,
    engine: &mut GraphEngine,
) -> NodeId {
    let name = child_name(node, source);
    let func_id = add_node(
        engine,
        NodeKind::Function,
        &name,
        &format!("{name} in {file_path}"),
    );
    set_byte_meta(engine, &func_id, file_path, node);
    engine.add_edge(Edge::new(parent_id, func_id, EdgeKind::Contains));

    if let Some(body) = node.child_by_field_name("body") {
        collect_py_calls(body, source, func_id, engine);
    }

    func_id
}

fn process_py_class(
    node: tree_sitter::Node,
    source: &str,
    file_path: &str,
    parent_id: NodeId,
    engine: &mut GraphEngine,
    func_name_to_ids: &mut HashMap<String, Vec<NodeId>>,
) -> NodeId {
    let name = child_name(node, source);
    let class_id = add_node(
        engine,
        NodeKind::Class,
        &name,
        &format!("class {name} in {file_path}"),
    );
    set_byte_meta(engine, &class_id, file_path, node);
    engine.add_edge(Edge::new(parent_id, class_id, EdgeKind::Contains));

    if let Some(body) = node.child_by_field_name("body") {
        for i in 0..body.named_child_count() {
            let Some(child) = body.named_child(i) else { continue };
            match child.kind() {
                "function_definition" => {
                    let fid = process_py_function(child, source, file_path, class_id, engine);
                    func_name_to_ids
                        .entry(child_name(child, source))
                        .or_default()
                        .push(fid);
                }
                "class_definition" => {
                    let cid = process_py_class(child, source, file_path, class_id, engine, func_name_to_ids);
                    func_name_to_ids
                        .entry(child_name(child, source))
                        .or_default()
                        .push(cid);
                }
                _ => {}
            }
        }
    }

    class_id
}

fn process_py_import(
    node: tree_sitter::Node,
    source: &str,
    file_id: NodeId,
    engine: &mut GraphEngine,
) {
    let module_name: Option<String> = match node.kind() {
        "import_statement" => node
            .named_child(0)
            .map(|n| n.utf8_text(source.as_bytes()).unwrap_or("").to_string()),
        "import_from_statement" => node
            .child_by_field_name("module_name")
            .map(|n| n.utf8_text(source.as_bytes()).unwrap_or("").to_string()),
        _ => None,
    };

    if let Some(name) = module_name {
        let top_module = name.split('.').next().unwrap_or(&name);
        if top_module.is_empty() || is_stdlib_module(top_module) {
            return;
        }

        let target_id = find_node_by_name(engine, top_module, &NodeKind::Module)
            .unwrap_or_else(|| {
                add_node(
                    engine,
                    NodeKind::Module,
                    top_module,
                    &format!("imported module {top_module}"),
                )
            });
        engine.add_edge(Edge::new(file_id, target_id, EdgeKind::Imports));
    }
}

/// Recursively walk a Python AST node for `call` sites, creating `Calls` edges.
fn collect_py_calls(
    node: tree_sitter::Node,
    source: &str,
    caller_id: NodeId,
    engine: &mut GraphEngine,
) {
    for i in 0..node.named_child_count() {
        let Some(child) = node.named_child(i) else { continue };
        if child.kind() == "call" {
            let call_target = child
                .child_by_field_name("function")
                .map(|f| f.utf8_text(source.as_bytes()).unwrap_or("").to_string())
                .unwrap_or_default();
            if !call_target.is_empty() && !is_builtin(call_target.as_str()) {
                if let Some(target_id) =
                    find_node_by_name(engine, call_target.as_str(), &NodeKind::Function)
                        .or_else(|| {
                            find_node_by_name(engine, call_target.as_str(), &NodeKind::Class)
                        })
                {
                    engine.add_edge(Edge::new(caller_id, target_id, EdgeKind::Calls));
                }
            }
        }
        collect_py_calls(child, source, caller_id, engine);
    }
}

// ── TypeScript helpers ───────────────────────────────────────────────────

fn process_ts_class_body(
    body: tree_sitter::Node,
    source: &str,
    file_path: &str,
    class_id: NodeId,
    engine: &mut GraphEngine,
    func_name_to_ids: &mut HashMap<String, Vec<NodeId>>,
) {
    for i in 0..body.named_child_count() {
        let Some(child) = body.named_child(i) else { continue };
        if child.kind() == "method_definition" {
            let name = child_name(child, source);
            let mid = add_node(engine, NodeKind::Function, &name, &format!("method {name} in {file_path}"));
            set_byte_meta(engine, &mid, file_path, child);
            engine.add_edge(Edge::new(class_id, mid, EdgeKind::Contains));
            func_name_to_ids.entry(name).or_default().push(mid);
        }
    }
}

// ── shared helpers ───────────────────────────────────────────────────────

fn add_node(engine: &mut GraphEngine, kind: NodeKind, name: &str, summary: &str) -> NodeId {
    let mut node = Node::new(kind, summary);
    node.name = name.to_string();
    let id = node.id;
    engine.add_node(node);
    id
}

fn child_name(node: tree_sitter::Node, source: &str) -> String {
    node.child_by_field_name("name")
        .and_then(|n| n.utf8_text(source.as_bytes()).ok())
        .map_or_else(|| String::from("unknown"), String::from)
}

fn set_byte_meta(engine: &mut GraphEngine, id: &NodeId, file_path: &str, node: tree_sitter::Node) {
    if let Some(n) = engine.get_node_mut(id) {
        n.set_meta("file", MetaValue::Text(file_path.to_string()));
        #[allow(clippy::cast_precision_loss)]
        {
            n.set_meta("start_byte", MetaValue::Number(node.start_byte() as f64));
            n.set_meta("end_byte", MetaValue::Number(node.end_byte() as f64));
        }
    }
}

fn find_node_by_name(engine: &GraphEngine, name: &str, kind: &NodeKind) -> Option<NodeId> {
    engine
        .all_nodes()
        .iter()
        .find(|n| &n.kind == kind && n.name == name)
        .map(|n| n.id)
}

fn is_skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "node_modules"
            | "vendor"
            | "third_party"
            | "build"
            | "dist"
            | "target"
            | "__pycache__"
            | ".pytest_cache"
            | ".mypy_cache"
            | ".tox"
            | "site-packages"
            | "venv"
            | ".venv"
            | "env"
            | ".env"
    )
}

fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "print"
            | "len"
            | "range"
            | "int"
            | "str"
            | "float"
            | "bool"
            | "list"
            | "dict"
            | "set"
            | "tuple"
            | "type"
            | "isinstance"
            | "super"
            | "enumerate"
            | "zip"
            | "map"
            | "filter"
            | "open"
            | "getattr"
            | "setattr"
            | "hasattr"
            | "delattr"
            | "__import__"
            | "abs"
            | "all"
            | "any"
            | "bin"
            | "chr"
            | "divmod"
            | "format"
            | "hex"
            | "id"
            | "input"
            | "iter"
            | "max"
            | "min"
            | "next"
            | "oct"
            | "ord"
            | "pow"
            | "repr"
            | "reversed"
            | "round"
            | "slice"
            | "sorted"
            | "sum"
            | "vars"
    )
}

fn is_stdlib_module(name: &str) -> bool {
    matches!(
        name,
        "os"
            | "sys"
            | "re"
            | "json"
            | "math"
            | "time"
            | "datetime"
            | "collections"
            | "itertools"
            | "functools"
            | "typing"
            | "abc"
            | "io"
            | "pathlib"
            | "logging"
            | "unittest"
            | "subprocess"
            | "threading"
            | "multiprocessing"
            | "hashlib"
            | "base64"
            | "uuid"
            | "random"
            | "statistics"
            | "csv"
            | "xml"
            | "html"
            | "urllib"
            | "http"
            | "socket"
            | "ssl"
            | "email"
            | "traceback"
            | "warnings"
            | "shutil"
            | "tempfile"
            | "glob"
            | "fnmatch"
            | "argparse"
            | "configparser"
            | "dataclasses"
            | "enum"
            | "textwrap"
            | "string"
            | "struct"
            | "pickle"
            | "sqlite3"
            | "copy"
            | "pprint"
            | "ast"
            | "inspect"
            | "concurrent"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_python_repo() -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/validators.py"),
            r#"
import re
from typing import List

def is_valid_username(name: str) -> bool:
    return bool(re.match(r'^[\w.@+-]+$', name))

class UsernameValidator:
    def __init__(self, min_len: int = 3):
        self.min_len = min_len

    def validate(self, name: str) -> bool:
        if len(name) < self.min_len:
            return False
        return is_valid_username(name)
"#,
        )
        .unwrap();
        std::fs::write(root.join("README.md"), "# Test Project\n").unwrap();
        td
    }

    fn fake_rust_repo() -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/lib.rs"),
            r#"
pub struct Validator {
    min_len: usize,
}

impl Validator {
    pub fn new(min_len: usize) -> Self {
        Self { min_len }
    }

    pub fn validate(&self, name: &str) -> bool {
        name.len() >= self.min_len
    }
}

pub fn is_valid_username(name: &str) -> bool {
    !name.is_empty()
}
"#,
        )
        .unwrap();
        td
    }

    // ── Python tests ─────────────────────────────────────────────────

    #[test]
    fn build_creates_file_and_module_nodes() {
        let repo = fake_python_repo();
        let builder = AstGraphBuilder::default();
        let engine = builder.build(repo.path());

        let files = engine.nodes_by_kind(&NodeKind::File);
        assert_eq!(files.len(), 1, "should have one File node");
        assert!(
            files[0].name.ends_with("validators.py"),
            "File name should end with validators.py, got: {}",
            files[0].name
        );

        let modules = engine.nodes_by_kind(&NodeKind::Module);
        assert!(!modules.is_empty(), "should have at least one Module node");
    }

    #[test]
    fn build_extracts_function_and_class_nodes() {
        let repo = fake_python_repo();
        let builder = AstGraphBuilder::default();
        let engine = builder.build(repo.path());

        let functions = engine.nodes_by_kind(&NodeKind::Function);
        let func_names: Vec<&str> = functions.iter().map(|n| n.name.as_str()).collect();
        assert!(func_names.contains(&"is_valid_username"), "should find top-level function, got: {func_names:?}");
        assert!(func_names.contains(&"__init__"), "should find __init__ method, got: {func_names:?}");
        assert!(func_names.contains(&"validate"), "should find validate method, got: {func_names:?}");

        let classes = engine.nodes_by_kind(&NodeKind::Class);
        let class_names: Vec<&str> = classes.iter().map(|n| n.name.as_str()).collect();
        assert!(class_names.contains(&"UsernameValidator"), "should find UsernameValidator class, got: {class_names:?}");
    }

    #[test]
    fn build_creates_contains_edges() {
        let repo = fake_python_repo();
        let builder = AstGraphBuilder::default();
        let engine = builder.build(repo.path());

        let modules = engine.nodes_by_kind(&NodeKind::Module);
        let module_id = modules
            .iter()
            .find(|n| n.name.starts_with("module:src/"))
            .map(|n| n.id)
            .expect("should find module node");

        let neighbors = engine.get_neighbors(&module_id);
        let contains_count = neighbors
            .iter()
            .filter(|(_, e)| e.kind == EdgeKind::Contains)
            .count();
        assert!(contains_count >= 2, "module should contain at least class + function, got {contains_count}");
    }

    #[test]
    fn build_skips_non_source_files() {
        let repo = fake_python_repo();
        let builder = AstGraphBuilder::default();
        let engine = builder.build(repo.path());
        let files = engine.nodes_by_kind(&NodeKind::File);
        assert!(files.iter().all(|f| !f.name.contains("README")), "README.md should be skipped");
    }

    #[test]
    fn extract_chunks_returns_source_text() {
        let repo = fake_python_repo();
        let builder = AstGraphBuilder::default();
        let engine = builder.build(repo.path());
        let chunks = builder.extract_chunks(&engine, repo.path());

        assert!(!chunks.is_empty(), "should have chunks for functions/classes");
        let func_chunk = chunks.iter().find(|(_, c)| c.text.contains("def is_valid_username"));
        assert!(func_chunk.is_some(), "should have a chunk for is_valid_username function");
        let (_, chunk) = func_chunk.unwrap();
        assert!(chunk.text.contains("def is_valid_username"));
        assert!(chunk.path.ends_with("validators.py"));
        assert!(chunk.line_start >= 1, "line_start should be 1-indexed");
        assert!(chunk.line_end > chunk.line_start, "line range should be valid");
    }

    #[test]
    fn empty_repo_returns_empty_graph() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("src")).unwrap();
        std::fs::write(td.path().join("src/readme.txt"), "hello").unwrap();
        let builder = AstGraphBuilder::default();
        let engine = builder.build(td.path());
        assert_eq!(engine.node_count(), 0, "empty repo should have no nodes");
    }

    // ── Rust tests ───────────────────────────────────────────────────

    #[test]
    fn rust_build_extracts_functions_and_structs() {
        let repo = fake_rust_repo();
        let builder = AstGraphBuilder::default();
        let engine = builder.build(repo.path());

        let functions = engine.nodes_by_kind(&NodeKind::Function);
        let func_names: Vec<&str> = functions.iter().map(|n| n.name.as_str()).collect();
        assert!(func_names.contains(&"is_valid_username"), "should find top-level fn, got: {func_names:?}");
        assert!(func_names.contains(&"new"), "should find impl method new, got: {func_names:?}");
        assert!(func_names.contains(&"validate"), "should find impl method validate, got: {func_names:?}");

        let classes = engine.nodes_by_kind(&NodeKind::Class);
        let class_names: Vec<&str> = classes.iter().map(|n| n.name.as_str()).collect();
        assert!(class_names.contains(&"Validator"), "should find Validator struct, got: {class_names:?}");
    }

    #[test]
    fn lang_from_extension() {
        assert_eq!(Lang::from_extension("py"), Some(Lang::Python));
        assert_eq!(Lang::from_extension("rs"), Some(Lang::Rust));
        assert_eq!(Lang::from_extension("ts"), Some(Lang::TypeScript));
        assert_eq!(Lang::from_extension("tsx"), Some(Lang::TypeScript));
        assert_eq!(Lang::from_extension("go"), Some(Lang::Go));
        assert_eq!(Lang::from_extension("java"), Some(Lang::Java));
        assert_eq!(Lang::from_extension("md"), None);
        assert_eq!(Lang::from_extension("txt"), None);
    }
}
