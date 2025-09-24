# ForecastAI-Crypto-Bridging-Blockchain-Trends-Social-Whispers

## Project Metadata
### Authors
- **Team:** g202401140 - Waseem Mohamad
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Write 1-2 technical paragraphs (feel free to add images if you would like).

The paper that I have chosen to work on is "Deep learning and NLP in cryptocurrency forecasting: Integrating financia, blockchain, and social media data" by Gurgul, Lessmann and Härdle. This paper is inspired by the recent events in the cryptocurrency world and is related to cryptocurrency forecasting using application of latest techniques.

With the rapid growth of cryptocurrency markets over the last decade, it has become a center of attention for both Financial Technology (FinTech) enthusiasts and computer scientists due to their high volatility and sensitivity to multifaceted drivers. Contrary to traditional assets, the prices of cryptocurrency get affected by blockchain activity, social media sentiments, and influtential individuals' opinions while also getting some influence by financial indicators. The applications of Artificial Intelligence (AI) like Machine Learning (ML) approaches have gained notoriety for this domain, specifically Deep Learning (DL) techniques, as they can model complicated nonlinear dependencies and exploit the highly dimensional data.
   
The paper mentions that the recent studies have surpassed the univariate price series to multimodal approaches that integrate heterogeneous signals, such as blockchain data and textual sentiments (social media). Especially, techniques like transformers and advanced Natural Language Processing (NLP) models have demonstrated promising capabilities in extracting nuanced features from text and merging them with numerial predictors. With these state-of-the-art techniques, many opportunities open up to design more robust forecasting frameworks that better adapt to volatile regimes and market shocks.

## Problem Statement
Write 1-2 technical paragraphs (feel free to add images if you would like).

The paper states that the traditional predicting models often struggle to capture the complex drivers of cryptocurrency markets as they solely depend on the historical prices and volume data of assets. Furthermore, the researchers convey that even existing multimodal techniques face pivotal constraints. The paper integrates financial, blockchain and social media data but also employs comparatively conventional NLP methods and basic fusion strategies. In conclusion, cross modal interactions may be underutilized and external macroeconomic influences are ignored.

This further leads to models that may perform well under stable conditions but lack robustness during turbulent events such as regulatory announcements, global macro shocks or even sudden shifts in investor sentiment. Therefore, according to the paper, a need for forecasting frameworks is required that combine richer data, leverage advanced architectures and provide resilience across different market conditions or environments.

## Application Area and Project Domain
Write 1-2 technical paragraphs (feel free to add images if you would like).

The main application area of this paper is "Forecasting" the assets prices, mainly cryptocurrencies, which is a field of high relevance to crypto enthusiasts, investors, traders and even policymakers. In general, the domain lies at the crosspath of DL, FinTech and multimodal daata science. Also, the paper leverages blockchain analytics, social media NLP, and financial time series modeling to sow the seeds for forecasts that are both technically rigorous and practically useful. Other than the crypto field, the methods explored in this paper contribute to the wider body of research on multimodal forecasting, highlighting how the heterogeneous data sources can be combined to improve the predictions in complex and dynamic systems.

## What is the paper trying to do, and what are you planning to do?
Write 1-2 technical paragraphs (feel free to add images if you would like).

The paper tries to demonstrate that incorporating blockchain and social sentiment alongside financial dataa improves the predictive accuracy. The models mentioned in the paper use DL and NLP to integrate the multimodal signals while also showing the benefits of a holistic approach.

I will try to replicate the work done by the authors and extend on its research and implmentation in following ways:


# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
