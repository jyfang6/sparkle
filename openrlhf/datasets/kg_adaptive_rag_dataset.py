from torch.utils.data import Dataset


class KGAdaptiveRAGDataset(Dataset):

    def __init__(self, data: Dataset):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        question = self.data[idx]["question"]
        label = self.data[idx]["answers"]
        # return question, label 
        return {
            "question": question,
            "label": label,
        }
    
def custom_collate_fn(batch):

    questions = [example["question"] for example in batch]
    labels = [example["label"] for example in batch]
    return questions, labels
