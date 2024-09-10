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
### Environment setup
```
conda create --name autosurvey python=3.9 -y
conda activate autosurvey

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric
pip install backoff
pip install scholarly
pip install fuzzywuzzy
```
---
### Process dataset
```
python -m src.dataset.build_graph
python -m src.dataset.update_abstract
```
