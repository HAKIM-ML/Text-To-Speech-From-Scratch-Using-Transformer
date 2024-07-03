# Transformer-Based Text-to-Speech (TTS) from Scratch

This repository contains a Transformer-based Text-to-Speech (TTS) model built from scratch using PyTorch. The model aims to generate mel spectrograms from text inputs, which can be further processed to generate audio.

## Features

- **Custom Transformer Blocks:** Implementation of custom encoder and decoder blocks with self-attention and feedforward layers.
- **Encoder and Decoder Pre-Nets:** Pre-net layers to process inputs before passing them to the main Transformer blocks.
- **Post-Net:** A network to refine the output mel spectrograms from the decoder.
- **Inference Mode:** Inference function to generate mel spectrograms from text inputs with stop token prediction.
- **DataLoader Testing:** Function to test the model using a DataLoader.

## Model Architecture

- **EncoderBlock:** Comprises self-attention and feedforward networks with layer normalization and residual connections.
- **DecoderBlock:** Includes self-attention, encoder-decoder attention, and feedforward networks with layer normalization and residual connections.
- **EncoderPreNet:** Processes input text with embedding, linear transformations, and convolutional layers.
- **DecoderPreNet:** Processes mel spectrograms with linear transformations and dropout.
- **PostNet:** Refines the output of the decoder with a series of convolutional layers.
- **TransformerTTS:** Combines all the above components to generate mel spectrograms from text.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- Pandas
- tqdm

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/HAKIM-ML/Text-To-Speech-From-Scratch-Using-Transformer.git
    cd Text-To-Speech-From-Scratch-Using-Transformer
    ```



### Usage

#### Training

1. Prepare your dataset in a CSV file format.
2. Define your hyperparameters (hp) in a separate file or directly in your script.
3. Initialize and train the model using a DataLoader.

#### Inference

1. Use the `inference` method of the `TransformerTTS` class to generate mel spectrograms from text inputs.
2. Example inference script:

    ```python
    model = TransformerTTS().cuda()
    text = text_to_seq("Hello, world.").unsqueeze(0).cuda()
    mel_postnet, stop_token = model.inference(text, stop_token_threshold=1e3)
    print("mel_postnet:", mel_postnet.shape)
    print("stop_token:", stop_token.shape)
    ```

### Testing with DataLoader

Use the `test_with_dataloader` function to test the model with a DataLoader:
```python
def test_with_dataloader():
    df = pd.read_csv(hp.csv_path)
    dataset = TextMelDataset(df)  
    loader = torch.utils.data.DataLoader(
        dataset, 
        num_workers=1, 
        shuffle=False,
        sampler=None, 
        batch_size=4,
        pin_memory=True, 
        drop_last=True,       
        collate_fn=text_mel_collate_fn
    )

    model = TransformerTTS().cuda()
    
    for batch in loader:
        text_padded, text_lengths, mel_padded, mel_lengths, stop_token_padded = batch
        text_padded = text_padded.cuda()
        text_lengths = text_lengths.cuda()
        mel_padded = mel_padded.cuda()
        mel_lengths = mel_lengths.cuda()
        stop_token_padded = stop_token_padded.cuda()    

        post, mel, stop_token = model(text_padded, text_lengths, mel_padded, mel_lengths)
        print("post:", post.shape) 
        print("mel:", mel.shape) 
        print("stop_token:", stop_token.shape)
        break
