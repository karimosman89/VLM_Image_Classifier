from PIL import Image
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def classify(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_descriptions = ["a dog", "a cat", "a car", "a person"]
    text_tokens = clip.tokenize(text_descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    return text_descriptions[similarity.argmax().item()]

if __name__ == "__main__":
    print("Predicted:", classify("example.jpg"))
