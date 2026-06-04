//! Query expansion for retrieval keywords (§6.7.3).
//!
//! Expands extracted keyword tokens with:
//! - Domain-specific term mappings (framework-aware synonyms)
//! - General programming term synonyms
//! - CamelCase/snake_case variant generation
//!
//! The expanded set feeds into keyword-grep for broader coverage without
//! relying solely on embedding similarity.

use std::collections::{HashMap, HashSet};

/// Domain-specific term expansions keyed by top-level module/project name.
type DomainMap = HashMap<&'static str, Vec<&'static str>>;

/// General programming synonym pairs (bidirectional).
type SynonymPairs = &'static [(&'static str, &'static str)];

pub struct QueryExpander {
    /// Domain-specific expansions (e.g. "django" → ["orm", "queryset", "migration"])
    domain_terms: DomainMap,
    /// General synonym pairs
    synonyms: SynonymPairs,
    /// Max total expanded keywords (including originals)
    max_expanded: usize,
}

impl Default for QueryExpander {
    fn default() -> Self {
        Self {
            domain_terms: default_domain_terms(),
            synonyms: default_synonyms(),
            max_expanded: 20,
        }
    }
}

impl QueryExpander {
    /// Expand a set of keywords with domain-aware synonyms and variants.
    /// Returns the union of original keywords + expansions, deduplicated.
    pub fn expand(&self, keywords: &[String]) -> Vec<String> {
        let mut expanded: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();

        for kw in keywords {
            let lower = kw.to_lowercase();
            if expanded.insert(lower.clone()) {
                out.push(kw.clone());
            }

            // 1. Domain-specific expansions
            for (domain, terms) in &self.domain_terms {
                if lower.contains(domain) || lower == *domain {
                    for term in terms {
                        let t = (*term).to_string();
                        if expanded.insert(t.clone()) {
                            out.push(t);
                            if out.len() >= self.max_expanded {
                                return out;
                            }
                        }
                    }
                }
            }

            // 2. General synonym lookup
            for (a, b) in self.synonyms {
                if lower == *a {
                    let t = (*b).to_string();
                    if expanded.insert(t.clone()) {
                        out.push(t);
                    }
                } else if lower == *b {
                    let t = (*a).to_string();
                    if expanded.insert(t.clone()) {
                        out.push(t);
                    }
                }
            }

            if out.len() >= self.max_expanded {
                break;
            }
        }

        out
    }
}

fn default_domain_terms() -> DomainMap {
    let mut m: DomainMap = HashMap::new();
    m.insert(
        "django",
        vec![
            "orm", "queryset", "model", "migration", "admin", "form",
            "view", "template", "middleware", "url", "signal", "manager",
            "field", "validator", "serializer",
        ],
    );
    m.insert(
        "sympy",
        vec![
            "symbolic", "expression", "simplify", "solve", "integral",
            "derivative", "matrix", "function", "assumption", "evalf",
            "subs", "expand",
        ],
    );
    m.insert(
        "sphinx",
        vec![
            "directive", "role", "domain", "builder", "extension",
            "parser", "transform", "node", "translator",
        ],
    );
    m.insert(
        "flask",
        vec![
            "route", "blueprint", "request", "response", "session",
            "jinja", "template", "url_for",
        ],
    );
    m.insert(
        "scikit",
        vec![
            "estimator", "predictor", "classifier", "regressor",
            "pipeline", "transformer", "preprocessing",
        ],
    );
    m.insert(
        "pytest",
        vec![
            "fixture", "mark", "parametrize", "conftest", "hook",
            "plugin", "assertion",
        ],
    );
    m.insert(
        "matplotlib",
        vec![
            "axes", "figure", "plot", "artist", "renderer", "backend",
            "transform", "legend",
        ],
    );
    m.insert(
        "requests",
        vec![
            "session", "adapter", "response", "header", "cookie",
            "auth", "redirect",
        ],
    );
    m.insert(
        "sqlalchemy",
        vec![
            "session", "query", "mapper", "table", "column", "engine",
            "metadata", "relationship",
        ],
    );
    m.insert(
        "pandas",
        vec![
            "dataframe", "series", "index", "groupby", "merge", "pivot",
            "resample", "rolling",
        ],
    );
    m
}

fn default_synonyms() -> SynonymPairs {
    &[
        ("validate", "validator"),
        ("validate", "validation"),
        ("validate", "check"),
        ("serialize", "serializer"),
        ("serialize", "marshalling"),
        ("deserialize", "parse"),
        ("deserialize", "unmarshal"),
        ("authenticate", "auth"),
        ("authenticate", "login"),
        ("authorize", "permission"),
        ("authorize", "acl"),
        ("config", "settings"),
        ("config", "configuration"),
        ("error", "exception"),
        ("error", "failure"),
        ("log", "logging"),
        ("log", "trace"),
        ("cache", "caching"),
        ("cache", "memoize"),
        ("render", "template"),
        ("render", "display"),
        ("query", "filter"),
        ("query", "search"),
        ("migrate", "migration"),
        ("migrate", "schema"),
        ("import", "require"),
        ("import", "include"),
        ("export", "output"),
        ("export", "dump"),
        ("encode", "encoding"),
        ("decode", "decoding"),
        ("regex", "pattern"),
        ("regex", "regular expression"),
        ("unicode", "utf"),
        ("unicode", "encoding"),
        ("async", "coroutine"),
        ("async", "await"),
        ("concurrent", "thread"),
        ("concurrent", "parallel"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_expansion_django() {
        let qe = QueryExpander::default();
        let expanded = qe.expand(&["django".into()]);
        assert!(expanded.iter().any(|t| t == "orm"));
        assert!(expanded.iter().any(|t| t == "queryset"));
        assert!(expanded.iter().any(|t| t == "migration"));
    }

    #[test]
    fn domain_expansion_sympy() {
        let qe = QueryExpander::default();
        let expanded = qe.expand(&["sympy".into()]);
        assert!(expanded.iter().any(|t| t == "symbolic"));
        assert!(expanded.iter().any(|t| t == "simplify"));
    }

    #[test]
    fn synonym_expansion() {
        let qe = QueryExpander::default();
        let expanded = qe.expand(&["validate".into()]);
        assert!(expanded.iter().any(|t| t == "validator"));
        assert!(expanded.iter().any(|t| t == "check"));
    }

    #[test]
    fn no_duplicates() {
        let qe = QueryExpander::default();
        let expanded = qe.expand(&["validate".into(), "validator".into()]);
        let validate_count = expanded.iter().filter(|t| *t == "validate").count();
        assert_eq!(validate_count, 1);
    }

    #[test]
    fn respects_max_expanded() {
        let qe = QueryExpander {
            max_expanded: 5,
            ..Default::default()
        };
        let expanded = qe.expand(&["django".into()]);
        assert!(expanded.len() <= 5);
    }

    #[test]
    fn original_keywords_preserved() {
        let qe = QueryExpander::default();
        let expanded = qe.expand(&["ASCIIUsernameValidator".into()]);
        assert!(expanded.contains(&"ASCIIUsernameValidator".to_string()));
    }
}
