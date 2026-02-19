use notectl_core::config::Config;
use std::path::PathBuf;
use std::sync::Arc;

pub use notectl_daily_notes::{
    DailyNoteCapability, GetDailyNoteOperation, SearchDailyNotesOperation,
};
pub use notectl_files::{FileCapability, ListFilesOperation, ReadFilesOperation};
pub use notectl_outline::{
    GetOutlineOperation, GetSectionOperation, OutlineCapability, SearchHeadingsOperation,
};
pub use notectl_tags::{
    ExtractTagsOperation, ListTagsOperation, SearchByTagsOperation, TagCapability,
};
pub use notectl_tasks::{SearchTasksOperation, TaskCapability};

/// Registry for managing capabilities
pub struct CapabilityRegistry {
    task_capability: Arc<TaskCapability>,
    tag_capability: Arc<TagCapability>,
    file_capability: Arc<FileCapability>,
    daily_note_capability: Arc<DailyNoteCapability>,
    outline_capability: Arc<OutlineCapability>,
}

impl CapabilityRegistry {
    pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self {
        let file_capability = Arc::new(FileCapability::new(base_path.clone(), Arc::clone(&config)));
        let daily_note_capability = Arc::new(DailyNoteCapability::new(
            base_path.clone(),
            Arc::clone(&config),
            Arc::clone(&file_capability),
        ));

        Self {
            task_capability: Arc::new(TaskCapability::new(base_path.clone(), Arc::clone(&config))),
            tag_capability: Arc::new(TagCapability::new(base_path.clone(), Arc::clone(&config))),
            file_capability,
            daily_note_capability,
            outline_capability: Arc::new(OutlineCapability::new(base_path, Arc::clone(&config))),
        }
    }

    pub fn tasks(&self) -> Arc<TaskCapability> {
        Arc::clone(&self.task_capability)
    }

    pub fn tags(&self) -> Arc<TagCapability> {
        Arc::clone(&self.tag_capability)
    }

    pub fn files(&self) -> Arc<FileCapability> {
        Arc::clone(&self.file_capability)
    }

    pub fn daily_notes(&self) -> Arc<DailyNoteCapability> {
        Arc::clone(&self.daily_note_capability)
    }

    pub fn outline(&self) -> Arc<OutlineCapability> {
        Arc::clone(&self.outline_capability)
    }

    pub fn create_operations(&self) -> Vec<Arc<dyn notectl_core::operation::Operation>> {
        vec![
            Arc::new(SearchTasksOperation::new(self.tasks())),
            Arc::new(ExtractTagsOperation::new(self.tags())),
            Arc::new(ListTagsOperation::new(self.tags())),
            Arc::new(SearchByTagsOperation::new(self.tags())),
            Arc::new(ListFilesOperation::new(self.files())),
            Arc::new(ReadFilesOperation::new(self.files())),
            Arc::new(GetDailyNoteOperation::new(self.daily_notes())),
            Arc::new(SearchDailyNotesOperation::new(self.daily_notes())),
            Arc::new(GetOutlineOperation::new(self.outline())),
            Arc::new(GetSectionOperation::new(self.outline())),
            Arc::new(SearchHeadingsOperation::new(self.outline())),
        ]
    }
}
