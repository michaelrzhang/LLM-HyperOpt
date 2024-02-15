
# Large Language Models for Hyperparameter Optimization

Michael R. Zhang, Nishkrit Desai, Juhan Bae, Jonathan Lorraine, Jimmy Ba

We explore the use of foundational large language models (LLMs) in hyperparameter optimization (HPO).  Hyperparameters are critical in determining the effectiveness of machine learning models, yet their optimization often relies on manual approaches in limited budget settings. By prompting LLMs with dataset and model descriptions, we develop a methodology where LLMs suggest hyperparameter configurations, which are iteratively refined based on model performance. Our empirical evaluations on standard benchmarks reveal that LLMs, within constrained search budgets, can match or outperform traditional HPO methods like Bayesian optimization across eight datasets and four different machine learning models. Furthermore, we propose to treat the code specifying our model as a hyperparameter, which the LLM outputs, which affords greater flexibility than existing HPO approaches. 

*tl;dr*: You can just ask large language models (LLMs) which hyperparameters to use, and it works pretty well!


You can find code we use to run 2D experiments in `run_toy_fns.ipynb`. More code is coming soon (in the next week).

### Cite 
If you find our work useful, please cite as:
```
@article{zhang2023using,
  title={Using Large Language Models for Hyperparameter Optimization},
  author={Zhang, Michael R and Desai, Nishkrit and Bae, Juhan and Lorraine, Jonathan and Ba, Jimmy},
  journal={arXiv preprint arXiv:2312.04528},
  year={2023}
}
```