//! Ported from kent8192/engramoid `core/src/tracker/models.rs` (Phase 1).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceAction {
    pub id: String,
    pub session_id: String,
    pub tool_name: String,
    pub tool_input: String,
    pub tool_result_summary: String,
    pub timestamp: DateTime<Utc>,
    pub accessed_files: Vec<String>,
}

impl TraceAction {
    #[must_use]
    pub fn new(session_id: &str, tool_name: &str, tool_input: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            tool_name: tool_name.to_string(),
            tool_input: tool_input.to_string(),
            tool_result_summary: String::new(),
            timestamp: Utc::now(),
            accessed_files: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub project_path: String,
    pub task_description: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub task_succeeded: Option<bool>,
    pub summary: Option<String>,
    pub reward: Option<f64>,
    pub action_count: u64,
}

impl Session {
    #[must_use]
    pub fn new(project_path: &str, task_description: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            project_path: project_path.to_string(),
            task_description: task_description.to_string(),
            start_time: Utc::now(),
            end_time: None,
            task_succeeded: None,
            summary: None,
            reward: None,
            action_count: 0,
        }
    }

    pub fn end(&mut self, succeeded: bool, summary: &str, reward: f64) {
        self.end_time = Some(Utc::now());
        self.task_succeeded = Some(succeeded);
        self.summary = Some(summary.to_string());
        self.reward = Some(reward);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_action_creation() {
        let action = TraceAction::new("session-1", "Read", r#"{"file_path": "/src/main.rs"}"#);
        assert_eq!(action.session_id, "session-1");
        assert_eq!(action.tool_name, "Read");
        assert!(action.accessed_files.is_empty());
    }

    #[test]
    fn test_session_creation() {
        let session = Session::new("/projects/test", "Fix login bug");
        assert_eq!(session.project_path, "/projects/test");
        assert_eq!(session.task_description, "Fix login bug");
        assert!(session.end_time.is_none());
        assert!(session.reward.is_none());
    }

    #[test]
    fn test_session_end() {
        let mut session = Session::new("/projects/test", "Fix bug");
        session.end(true, "Fixed successfully", 0.85);
        assert!(session.end_time.is_some());
        assert_eq!(session.task_succeeded, Some(true));
        assert_eq!(session.reward, Some(0.85));
    }
}
