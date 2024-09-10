# SurveyGen

### Codes
#### 1. Data Processing:
- [x] Build Citation Network for Survery papers; 
- [x] Extract taxonomy trees;
- [x] Reference lables (sub-community) for subsection / sub-topic; _(Yuntong Hu)_

#### 2. Deep Clustering:
- [x] GNNs; 
- [ ] some

#### 3. LLM Generation:
- [ ] Prompt design;
- [x] Generator; 


### Paper Writing

- [ ] Introduction;
- [ ] Related Work;
- [ ] Problem Formulation;
- [ ] Experiment;
- [ ] Conclution;
- [ ] Appendix.

---
Process dataset
```
python -m src.dataset.build_graph
python -m src.dataset.update_abstract
```
