# Job Matching Script Using Combined TF-IDF and GloVe Embeddings

This Python script leverages a combination of TF-IDF vectorization and GloVe embeddings to match job descriptions from one dataset to another. It was designed to improve the accuracy of job matching by combining the strengths of both methods. This approach is particularly useful for NGOs and other organizations looking to align job descriptions across different systems or datasets.

## Features
- **TF-IDF Vectorization**: Captures the importance of words in the job descriptions relative to the entire dataset.
- **GloVe Embeddings**: Adds semantic meaning by using pre-trained word embeddings to represent job descriptions.
- **Combined Scoring**: Integrates both TF-IDF and GloVe for a comprehensive similarity score.
- **Top Matches**: Outputs the top 6 matches for each job description.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk

## Usage
1. **Install Dependencies**: Ensure that all required Python packages are installed.
    ```bash
    pip install pandas numpy scikit-learn nltk
    ```

2. **Download GloVe Embeddings**: Download the GloVe embeddings and place the file in your working directory.

3. **Edit the Script**: Replace the placeholders in the script with your specific file paths and any other necessary configurations.

4. **Run the Script**: Execute the script to perform job matching.
    ```bash
    python job-matching.py
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with any improvements or suggestions.

## Contact
For any questions or inquiries, please contact Drew Tompkins at dtompkins@vt.edu.


