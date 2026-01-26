import torch

def generate_pseudo_labels(model, input_ids):
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1)
        confidence, labels = probs.max(dim=-1)
    return labels, confidence
