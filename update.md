# 2023/09/09 update
- [ ] add .gitignore
- [ ] update the architecture
- [ ] update the cuda setup


- I have put all your previous code into the ./previous folder.
- `u_previous = torch.tensor(pd.read_csv('initial_0.csv').values[0:100,:], dtype=torch.float32, device=device) # Initial condition` I only selected 100 points for quickly testing. You can change these values as you like.
  

**What you should do:**
- Do the parameters-tuning.
- You can simply run in the terminal `python main.py --train --n_int 100 --epoch 1000 --device cuda` for using GPU for training. I didn't test the performance of GPU compared with CPU. You should carefully check about it. Plot the scale figure (computation time vs n_int) and figure out when you should use GPU or CPU. Using GPU is not always good.
- 


