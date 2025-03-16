# Final Project Summary

## Abstract / Objective / Motivation  

With the increasing role of AI in news automation and text processing, this study explores the effectiveness of fine-tuning generative models for semiconductor-related news. We analyze how GPT-2, a Transformer-based text generation model, performs before and after fine-tuning on real-world newsroom data.  

Our objective is to determine whether fine-tuning improves domain-specific accuracy and whether GPT-2 can generate coherent and relevant semiconductor-related content.


## Methodology  
### Pre-training & Fine-tuning

- Model Used: GPT-2  
- Dataset: TSMC-related news articles obtained via the Refinitiv API  
- Fine-tuning Approach: Training GPT-2 on newsroom data to adapt it for semiconductor-related content generation.

### Evaluation Metrics
- Perplexity (lower = better) → Measures how well the model predicts text.  
- BLEU Score (higher = better) → Measures similarity between generated text and real news articles.  

### Comparison
- Performance before and after fine-tuning on the same dataset  
- Analyzing whether fine-tuning helps GPT-2 generate more industry-specific news content  


## Data Source & Preprocessing
### Newsroom Dataset
- Source: Refinitiv API (LSEG News)  
- Content: 562 TSMC-related articles retrieved using automated news queries  
- Fields Extracted: Story text  
- Preprocessing Steps: Tokenization, cleaning, and formatting to ensure compatibility with GPT-2.  


## Results & Evaluations
### GPT-2 (Pretrained)
- Generates coherent but generic news-like text.  
- Lacks domain-specific semiconductor terminology.  
- Example Output:  
  "TSMC announced a new chip technology called the 'Cortex-Averaging' chip. The new chip is based on the Cortex-Averaging technology, which is used to process data from a computer's memory."

### GPT-2 (Fine-Tuned)
- Generates semiconductor-specific content but tends to repeat phrases.  
- Uses industry-relevant words but still has redundancy issues.  
- Example Output:  
  "TSMC announced a new chip technology called 'SEMICONDUCTOR' in March. The chip is a semiconductor process that is used to process semiconductor material, such as silicon, in a process called 'semiconductor lithography'."

### Metric Scores
Model | BLEU Score | Perplexity  
--- | --- | ---  
GPT-2 (Pretrained) | 0.0000 | N/A  
GPT-2 (Fine-Tuned) | 0.0071 | 52.80  

## Discussions
### Key Findings
- Fine-tuning slightly improves industry-specific content, but the model still struggles with repetitive phrasing.  
- Perplexity (52.80) remains high, indicating that more training data or better hyperparameters may be needed.  
- BLEU Score (0.0071) is very low, meaning the generated text does not closely match real news articles.  
- GPT-2 learns domain-specific vocabulary but fails to generate fully coherent and diverse news pieces.  

### Challenges & Limitations
- Repetitive Outputs → Model repeats words like "semiconductor process" multiple times.  
- Limited Training Data → Only 50 articles were used, which may not be enough for strong domain adaptation.  
- BLEU Score Issues → GPT-2’s output differs significantly from real newsroom articles, making BLEU an unreliable measure.  

## Conclusion
Fine-tuning GPT-2 does help adapt the model to semiconductor-related news, but the improvements are limited. The model learns industry terminology but struggles with coherence and diversity in generated news articles. Future improvements could include:
- Larger training datasets (more newsroom articles).  
- Training for more epochs for better adaptation.  
- Exploring alternative evaluation metrics (ROUGE, human assessment).  

Ultimately, GPT-2 is not ideal for newsroom content generation without significant enhancements, but it demonstrates potential for domain-specific AI-generated text.

## References / Citations
- Radford, A., Wu, J., Child, R., et al. (2019). "Language Models are Few-Shot Learners." OpenAI.  
- LSEG (Refinitiv API) Documentation: News Query API for financial and industry insights.  
