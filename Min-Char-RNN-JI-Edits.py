
# coding: utf-8

# In[432]:


"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy) and MODIFIED by JI
BSD License
"""
import numpy as np

# data I/O
#data = open('kafka_short.txt', 'r').read() # should be simple plain text file
data = open('hello.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
print(chars)
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))


# In[433]:


char_to_ix = { ch:i for i,ch in enumerate(chars)}
ix_to_char = { i:ch for i,ch in enumerate(chars)}
      
print(char_to_ix)
print(ix_to_char)


# In[434]:


# hyperparameters
hidden_size = 2 # size of hidden layer of neurons
seq_length = 2 # number of steps to unroll the RNN for
learning_rate = 1e-1

#Joey's execution parameters
sample_rate = 1 #samples every 'sample_rate' iterations
sample_length = 10
n_iterations = 25 #number of iterations, or number of training samples (a full forward & backward pass)
print(hidden_size,seq_length,learning_rate)


# In[435]:


# model parameters
#Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
#Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
#Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
Wxh = np.ones((hidden_size, vocab_size))*0.01 # input to hidden
Whh = np.ones((hidden_size, hidden_size))*0.01 # hidden to hidden
Why = np.ones((vocab_size, hidden_size))*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

#print(Wxh)
#print(Whh)
#print(Why)


# In[436]:


n, p = 0, 0
inputs=[]
for ch in data[p:p+seq_length]:
  inputs.append(char_to_ix[ch]) #same as code on next line
#inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
print(inputs)
print(targets)


# In[437]:


def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  loss_b4_log = 0
    
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    loss_b4_log += ps[t][targets[t],0]
    #yy = np.exp(ys[t])
    #yy_sum = np.sum(yy)
    #ps_value = yy/yy_sum
    #print("yy=",yy)
    #print("yy_sum=",yy_sum)
    #print("ps_value",ps_value)
    #print("\n")  
    #print("loss=", loss)
    #print("loss_b4_log=", loss_b4_log)
    #print("targets[t] =",targets[t])
    #print("ps[t] =",ps[t])
    #print("ps[t][targets[t]]=", ps[t][targets[t]])
    
    print("\n")  
    print("------Forward Pass for: n= %s and t= %s (View Transposed)------" % (n,t))
    print("t=",t)
    print("\n")    
    print("xs =", xs[t].T)
    print("\n")
    print("hs =", hs[t].T)
    print("\n")
    print("ys =", ys[t].T)
    print("\n")
    print("ps =", ps[t].T)
    print("\n")
    print("loss=", loss)
    print("loss_b4_log=", loss_b4_log)
    print("targets[t] =",targets[t])
    print("ps[t] =",ps[t])
    print("ps[t][targets[t]]=", ps[t][targets[t]])
 

      
  if n == 0:
    print("\n") 
    #print("\n-------------Weight Parameters-----------")  
    print("shape of xs", xs[1].shape)
    #print("\n")
    print("shape of hs", hs[1].shape)
    #print("\n")
    print("shape of Wxh", Wxh.shape)
    #print("\n")
    print("Weights of Wxh", (Wxh[:,:]))
    #print("\n")
    print("shape of Whh", Whh.shape)
    print("\n")
    print("Weights of Whh", (Whh))
    #print("\n")
    print("shape of Why", Why.shape)
    #print("\n")
    print("Weights of Why", (Why))
    print("shape of ys & ps", ys[1].shape)
    print("\n")
    
    #print("----------------Forward Pass------------")
    #print("\n")    
    #print("xs =", xs)
    #print("\n")
    #print("hs =", hs)
    #print("\n")
    #print("ys =", ys)
    #print("\n")
    #print("ps =", ps)
    #print("\n")
  
    #if n % sample_rate == 0:
    if False: # Round the probabilities matrix to make it more readable
      ps_rounded = ps
      for xx, yy in ps_rounded.items():
        #print(xx,len(yy))
        for aa in range(len(yy)):
            xyz = ps_rounded[xx][aa]
            xyz_rounded = round(xyz[0],2)
            #print(xyz_rounded)
            ps_rounded[xx][aa]=xyz_rounded
      print("ps_rounded =", ps_rounded)
      print("\n")

  

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
   
    print("\n")  
    print("-------  BACKWARD Pass for: n= %s and t= %s  -------" % (n,t))
    print("t=",t)
    print("\n")    
    print("dy =", dy)
    print("\n")
    print("dWhy += np.dot(dy, hs[t].T) \n dWhy =", dWhy)
    print("\n")
    print("dh =", dh)
    print("\n")
    print("dhraw =", dhraw)
    print("\n")
    
  #clip_for_count = 0
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    #clip_for_count += 1
    #print("clip_for_count=%s dparam=%s" % (clip_for_count, dparam))
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]



# In[438]:



def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    #print("ix=",ix)
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    #print("x=",x)
    ixes.append(ix)
  return ixes


#sample_ix = sample(hprev,inputs[0], 200)
#x = np.zeros((vocab_size, 1))
#ixes = []
#print("n=",n)
#print("hprev=", hprev)
#print("inputs[0]=", inputs[0])
#print("inputs=", inputs)
#print("sample_ix=",sample_ix)
#print("x=", x)
#print("ixes=" ,ixes)


# In[439]:


n, p, text_lap = 0, 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while n<= n_iterations : 
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
    text_lap += 1
    
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  #print("n=", n)
  #print("p=", p)
  #print('iter %d, smooth_loss: %f' % (n, smooth_loss)) # print progress
  #print("\n-------------------------------Next Iteration---------------------------------------------------------------------")
  #print("n=", n)
  #print("p=", p)

  # sample from the model now and then
  if n % sample_rate == 0:
  #if p == 0:
    print("\n-------------------------------Next Iteration---------------------------------------------------------------------")
    print("n=", n)
    print("p=", p)
    print("corpus lap=", text_lap)

    sample_ix = sample(hprev, inputs[0], sample_length)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    input_txt = ''.join(ix_to_char[ix] for ix in inputs)
    #print("n=", n)
    #print("p=", p)
    print("\n")
    print("First 20 values in hprev vector before Forward Pass:\n", hprev[0:20].ravel())
    #print("inputs[0]=", inputs[0])
    #print("Char =", ix_to_char[inputs[0]])
    #print("The first input=%d Which is a Char =%s" % (inputs[0],ix_to_char[inputs[0]]))
    print("\n")
    #print("inputs integers=", inputs)
    print("imput chars:'%s'" % input_txt)
    #print("sample integers= " , sample_ix)
    print("\n")
    print("sample of Predicted Chars:", txt)
    #print("-------Sample of Predicted Chars------\n %s " % (txt))
    #print("loss =", loss)
    

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % sample_rate == 0:
    print("\n")    
    print('For iter %d, loss: %f smooth_loss %f' % (n, loss, smooth_loss)) # print progress

  if n % sample_rate == 0: 
    if n == 0: 
        loss_start = loss
    if n == n_iterations:
        loss_at_last_iteration = loss
        smooth_loss_at_last_iteration = smooth_loss
        print("-------Loss Details------\n ")  
        print("Sequence Length", seq_length)
        print("Hidden Size", hidden_size)
        print("loss_start:", loss_start)
        print("loss_at last iteration:", loss_at_last_iteration)
        print("smooth loss_at last iteration:", smooth_loss_at_last_iteration)
  #print('iter %d, loss: %f' % (n, loss)) # print progress
  #print("\n-------------------------------Next Iteration---------------------------------------------------------------------")
    #print("\n loss=%s \n \n dWxh= %s \n \n dWhh= %s \n \n dWhy= %s \n \n dbh= %s \n \n dby=%s \n \n hs= %s" % (loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]))

  
  # perform parameter update with Adagrad
  zip_for_count = 0  
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    
    #zip_for_count += 1
    #print("\n zip_for_count= %s \n param=%s \n dparam=%s \n mem=%s \n" % (zip_for_count, param.ravel(),dparam.ravel(),mem.ravel()))
  p += seq_length # move data pointer
  n += 1 # iteration counter 


# In[440]:


#print(n)
#lossFun(inputs,targets,hprev)
a = [2, 4, 6]
print(a)
b = np.exp(a)
print(b)


# 
# """
# Joeys CODE
# 
# Joeys FWD PASS
# inputs,targets are both list of integers.
# hprev is Hx1 array of initial hidden state
# returns the loss
# """
# 
# xs, hs, ys, ps = {}, {}, {}, {}
# hprev = np.zeros((hidden_size,1)) # reset RNN memory
# hs[-1] = np.copy(hprev)
# loss = 0
#     
# # forward pass
# for t in range(len(inputs)):
#     xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
#     xs[t][inputs[t]] = 1
#     hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
#     ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
#     ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
#     loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
# 
# print("----------------Forward Pass------------")
# print("\n")    
# print("xs =", xs)
# print("\n")
# print("hs =", hs)
# print("\n")
# print("ys =", ys)
# print("\n")
# print("ps =", ps)
# print("\n")
# print("loss =", loss)
# print("\n")
# 
# """
# JOEYS Backward PASS - compute gradients
# inputs,targets are both list of integers.
# hprev is Hx1 array of initial hidden state
# returns the  gradients on model parameters, and last hidden state
# """
# 
# # backward pass: compute gradients going backwards
# dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
# dbh, dby = np.zeros_like(bh), np.zeros_like(by)
# dhnext = np.zeros_like(hs[0])
# 
# for t in reversed(range(len(inputs))):
#     dy = np.copy(ps[t])
#     dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
#     dWhy += np.dot(dy, hs[t].T)
#     dby += dy
#     dh = np.dot(Why.T, dy) + dhnext # backprop into h
#     dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
#     dbh += dhraw
#     dWxh += np.dot(dhraw, xs[t].T)
#     dWhh += np.dot(dhraw, hs[t-1].T)
#     dhnext = np.dot(Whh.T, dhraw)
# 
# print("----------------Backward Pass------------")
# print("\n")
# print("dy =", dy)
# print("\n")
# print("dWhy =", dWhy)
# print("\n")
# print("dWhh =", dWhh)
# print("\n")
# print("dWxh =", dWxh)
# 
