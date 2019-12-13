---
layout: default
title: Reinforcement Learning
parent: 1_Reinforcement_Learning
nav_order: 1

---

 Reinforcement Learning
{: .no_toc }

-- g_coord

```
def alter_coord(action, position, g_coord, dx=0.1, change_nodes=list(range(1,9))):
        
    if action==0:
        g_coord[int(2*change_nodes[position])]+=dx
        g_coord[int(2*change_nodes[position])+1]+=dx
    elif action==1:
        g_coord[int(2*change_nodes[position])]+=dx
        g_coord[int(2*change_nodes[position])+1]-=dx
    if action==2:
        g_coord[int(2*change_nodes[position])]-=dx
        g_coord[int(2*change_nodes[position])+1]+=dx
    elif action==3:
        g_coord[int(2*change_nodes[position])]-=dx
        g_coord[int(2*change_nodes[position])+1]-=dx    
    elif action==4:
        g_coord[int(2*change_nodes[position])+1]-=0
             
    return g_coorddnj
```

-- FE Model structure

```
return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
```

```
    pi=3.14159265   
    x = theta*pi/180
    C = math.cos(x)
    S = math.sin(x)
    w1 = A*C*C + 12*I*S*S/(L*L)
    w2 = A*S*S + 12*I*C*C/(L*L)
    w3 = (A-12*I/(L*L))*C*S
    w4 = 6*I*S/L
    w5 = 6*I*C/L
```

-- Policy Gradients

```
def discount_rewards(rewards, discount_rate=0.97):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards
```

```
def reward_(obs_,obs): 
    if obs_[-1]>obs[-1]:
        return 1
    else:
        return 0
```

-- Prediction

```
def predict(coord):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./policy4/pinjointed4.ckpt.meta')
        saver.restore(sess, "./policy4/pinjointed4.ckpt") 

        graph = tf.get_default_graph()
        outputs = graph.get_tensor_by_name("Y_proba:0") 
        X_ = graph.get_tensor_by_name("X_:0") 
        return obs,g_coord
```

```
def draw(coord,color,elcon):
    coord=coord.reshape(np.max(elcon)+1,2)
    plt.figure(figsize=(13,5))
    for item in elcon:
        plt.plot([coord[item[0]][0],coord[item[1]][0]],[coord[item[0]][1],coord[item[1]][1]],color=color)
       
    plt.show()    
```





