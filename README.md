# RWKV Based Music Generator

A 046211 - Deep Learning course final project at Technion's ECE faculty. <br>
<div style="text-align:center"><img src="./assets/canthelpfallinginlove-loveisintheair.gif">
<img src="./assets/music-box-of-sleep.gif"></div>

<br>

This project modifies an existing [LSTM-based model architecture](https://github.com/SudharshanShanmugasundaram/Music-Generation) designed to generate piano music and changes the architecture to use an RWKV model over the LSTM. <br>
The architecture comparison is based on the validation loss achieved and the quality of the music they generate.

## Agenda

- [Project Overview](#project-overview)
- [RWKV Music Generator Model Architecture](#rwkv-music-generator-model-architecture)
- [RWKV Model Compared To LSTM Based Modoel](#rwkv-music-generator-model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Project Overview

TL;DR - This project modifies an existing [LSTM-based model architecture](https://github.com/SudharshanShanmugasundaram/Music-Generation.git)  designed to generate piano music and changes the architecture to use an RWKV model over the LSTM. The RWKV model achieved better validation loss, and the generated music sounds better (at least to us:wink:).

### Baseline LSTM-based
#### LSTM - Long Short-Term Memory
* Long short-term memory (LSTM) is a type of recurrent cell that tries to preserve long-term information. The idea of LSTM was presented in 1997 but flourished in the age of deep learning.\n",
    * LSTM introduces a memory cell with the same shape as the hidden state, engineered to record additional information.
    * The memory is controlled by 3 main gates: 
        * **Input gate**: decides when to read data into the cell.
        * **Output gate**: outputs the entries from the cell.
        * **Forget gate**: a mechanism to reset the content of the cell.
    * These gates learn which information is relevant to forget or remember during the training process. The gates contain a sigmoid activation function.


### RWKV Music Generator Model Architecture


## Installation

Provide step-by-step instructions on how to install and set up the project. Include any prerequisites, dependencies, or environment setup required. You can use bullet points or code blocks to provide clear instructions.

## Usage

Provide examples and instructions on how to use the project. Explain the different components, modules, or functions available and how they can be utilized. Include code snippets or command examples if applicable.

## Acknowledgements

Acknowledge and give credit to any individuals, organizations, or open-source projects that have contributed to the project or influenced its development. Mention any external resources, references, or research papers that were used or referred to during the project.

## Contact

Provide contact information for the project maintainer or team. Include an email address or a link to a contact form. Optionally, you can include links to social media profiles or a dedicated project website.

