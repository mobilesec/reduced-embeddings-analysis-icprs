# Shrinking embeddings, not accuracy: Performance-preserving reduction of facial embeddings for complex face verification computations

This repository aims at replicating the results to be presented in:

- Philipp Hofer, Michael Roland, Philipp Schwarz, and René Mayrhofer: "Shrinking embeddings, not accuracy: Performance-preserving reduction of facial embeddings for complex face verification computations" in ICPRS '24: 14th International Conference on Pattern Recognition Systems (accepted)

In our publication we present various statistics using the code from this repository.

## Setup

- To compile the binary for this project, ensure that Nix is installed on your system. Once Nix is installed, open your terminal and execute the following command to build the binary: `nix build`. This command will automatically fetch the necessary dependencies and compile the source code into a runnable binary. The compiled binary will be stored in `./result/bin/reducedemb`.
- To run this application, you must specify which dataset to operate on by using the `--data` flag followed by the complexity of the dataset (`easy` for LFW and `hard` for CPLFW).
    - `--easy --lfwpath [path_to_lfw]`: Sets the path to the LFW dataset.
    - `--hard --cplfwpath [path_to_cplfw]`: Sets the path to the CPLFW dataset.
- Run the application by specifying the required dataset and its path, along with the action to perform on the dataset. Example: `./result/bin/reducedemb --data easy --lfwpath "/path/to/lfw" --action cache`
- Available actions:
    - `--action cache`: Caches the data modifying the records in place.
    - `--action extract-emb`: Extracts embeddings and updates records.
    - `--action truncate-embedding-size`: Truncates the size of embeddings to a fixed dimension.
    - `--action truncate-embedding-size-rel`: Truncates embedding sizes relatively.
    - `--action random-dimensions`: Randomly reduces dimensions to a specified subset.
    - `--action random-dimensions-full`: Applies dimensionality reduction to the entire dataset.
    - `--action best-elements-full`: Identifies and retains the most significant elements, requiring specification of the number.
    - `--action best-elements-greedy`: Similar to best-elements-full but uses a greedy algorithm for selection.
    - `--action heatmap`: Generates a heatmap from the data dimensions.
    - `--action quant`: Quantizes the dataset.
    - `--action proposed`: Executes a proposed action customized for specific requirements.
- For actions that require specifying the number of elements or dimensions, use: `--amount [number]`

## Acknowledgements

This work has been carried out within the scope of Digidow, the Christian Doppler Laboratory for Private Digital Authentication in the Physical World and has partially been supported by the LIT Secure and Correct Systems Lab. We gratefully acknowledge financial support by the Austrian Federal Ministry of Labour and Economy, the National Foundation for Research, Technology and Development, the Christian Doppler Research Association, 3 Banken IT GmbH, ekey biometric systems GmbH, Kepler Universitätsklinikum GmbH, NXP Semiconductors Austria GmbH & Co KG, Österreichische Staatsdruckerei GmbH, and the State of Upper Austria.

## License

This project is licensed under the EUPL, Version 1.2. For more details, see the [EUPL License](https://joinup.ec.europa.eu/software/page/eupl).
