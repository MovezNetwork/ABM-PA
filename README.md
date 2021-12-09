# ABM-PA

An agent-based model for physical activity behaviour in children and adolescents.
In the model agents' physical activity is influenced by peers’ behavior (social network influence) and affected by socio-economic conditions of the agent (environmental influence). Agents connect to each other via derived social connections in the nominated or communication network.

## Agents 
Children and adolescents in schoolclasses

## Attributes 
Social network: peer-nomination network or online communication network <br>
Physical activity level <br>
Environmental score (based on familty affluence scale) 

## Behavior 
The agent’s physical activity level is determined by:<br>
1. peers' physical activity level (j) and strength of connection:
![image](https://user-images.githubusercontent.com/78726753/144422632-d0872ea2-025a-45ec-bf2c-f23d6a3b5446.png)
2. environmental score **_env_**: 
![image](https://user-images.githubusercontent.com/78726753/144423954-70267100-b4e3-4ae1-8c06-6585d138af52.png)
3. Threshold for changing physical activty behavior **_T<sub>PAL</sub>_**(influence score should exceed the threshold):<br>
![image](https://user-images.githubusercontent.com/78726753/144424595-c28185ec-b819-4d2d-b714-7144de8df7fa.png)
4. Factor of increasing or decreasing physical activity **_I<sub>PAL</sub>_**:
![image](https://user-images.githubusercontent.com/78726753/144424833-cc15cb3b-b7ee-48a0-a076-6d304e00ce8d.png)


