# Introduction
With the prevalent use of Large Language Models (LLMs) like ChatGPT, the need for robust detection methods has become critical due to concerns about plagiarism, misinformation, and fake news. This paper compares traditional machine learning classifiers, zero-shot detection models, advanced techniques leveraging pre-trained neural networks, and watermark-based de-tection methods. A comparative study is conducted using a dataset of Quora-like question-answer pairs generated by both humans and GPT-4, focusing on three detection models: Lo-gistic Regression, DetectGPT, and Ghostbuster. The study highlights the challenges in creating adaptable and robust detection systems that can keep pace with evolving LLMs and emphasizes the importance of system generalization and the need for future research to focus on methods less dependent on training domains.

--- 

# Data Processing
- data_processing.py 
    - data cleaning of the Huggingface Quora dataset (toughdata/quora-question-answer-dataset)
    - filter the human-written answers with words between [300, 1000]
    - saved as "processed_data.csv"
- truncate_to_500.py
	- truncate the original data to 500 questions-answers
	- saved as "pruned_data.csv"
- gpt_text_generator.py
	- call "gpt-4-1106-preview" model to generate answers based on the following prompt
	- Prompt:
		Generate a Quora-like answer based on the question "<question>" with referring to 
		the answer "<answer>" and start the answer with the first 8 words of the reference 
		answer. The generated answer should have similar word amount of the reference answer.
	- saved as "gpt4_data.csv"
- word_counts.py
	- calculate the average word count for human-written answers and GPT4-generated answers
- text_similarity.py
	- calculate the mean of consine similarity of human-written answers and GPT4-generated answers
- topic_modeling_data.py
	- process the data to keep only the questions
	- use online topic modeling model jsLDA: In-browser topic modeling to find the topics of the data
- process_gpt4_text.py
	- mark the human answer with label "0", and GPT-4 generated answer with label "1"
	- only keep two columns "text" and "label"
	- saved as "gpt4_data_processed.csv"
