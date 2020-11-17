# Me learning neural networks

Hello this is very bad neural network to detect a dog or a cat. 

If you want to run through the training part yourselft with the images and everything get the dataset from <a href = "https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765">microsoft petImages</a>.
Else you can also just use the .pickle files where i have stored the compressed dataset which is already shuffled.
# Is it good?
No it isn't but it did give me a better understanding of how neural networks work.
Here you can the the graph of the cost and if i must say so myself it does look pretty nice.

<img src="https://github.com/4C4F4943/me_learning_neur_net/blob/main/cost_plot.png" width="40%" height="40%">

But then the thing everybody want to know the accuracy. After optimizing the learning rate and the iterations to a whopping <br>
120 000(yes the graph isn't the right one) i got a trian accuracy of about 80% and a test accuracy of about the same.

This is because i didn't have a good test dataset so i had to use the original one so basicaly it is kindeof bad...

Well hope this could help you in a way.

# further usage

if you want to test it with you're own images (i doubt you will) then do as follows in the <a href="https://github.com/4C4F4943/me_learning_neur_net/blob/main/load_in.py">load_in.py</a> where you use the already optimized weights(w.pickle) and bias.
```python
test_x = cv2.imread("[YOU'RE OWN IMAGE]",cv2.IMREAD_GRAYSCALE)
test_x = cv2.resize(test_x,(IMG_SIZE,IMG_SIZE))

```
