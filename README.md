Download Link: https://assignmentchef.com/product/solved-cpts-570-machine-learning-homework-3
<br>









<h1>Analytical Part (3 Percent)</h1>

<ol>

 <li>Suppose <em>x </em>= (<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,</em>··<em>,x<sub>d</sub></em>) and <em>z </em>= (<em>z</em><sub>1</sub><em>,z</em><sub>2</sub><em>,</em>···<em>,z<sub>d</sub></em>) be any two points in a high-dimensional space (i.e., <em>d </em>is very large).

  <ol>

   <li>Try to prove the following, where the right-hand side quantity represent the standardEuclidean distance.</li>

  </ol></li>

</ol>

(1)

<strong>Hint: </strong>Use Jensen’s inequality – If <em>X </em>is a random variable and <em>f </em>is a convex function, then <em>f</em>(E[<em>X</em>]) ≤ E[<em>f</em>(<em>X</em>)].

<ol>

 <li>We know that the computation of nearest neighbors is very expensive in the highdimensional space. Discuss how we can make use of the above property to make the nearest neighbors computation efficient?</li>

</ol>

<ol start="2">

 <li>We briefly discussed in the class about Locality Sensitive Hashing (LSH) algorithm to make the nearest neighbor classifier efficient. Please read the following paper and briefly summarize the key ideas as you understood:</li>

</ol>

Alexandr Andoni, Piotr Indyk: Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions. Communications of ACM 51(1): 117-122 (2008) <a href="https://people.csail.mit.edu/indyk/p117-andoni.pdf">http:// </a><a href="https://people.csail.mit.edu/indyk/p117-andoni.pdf">people.csail.mit.edu/indyk/p117-andoni.pdf</a>

<ol start="3">

 <li>We know that we can convert any decision tree into a set of if-then rules, where there is one rule per leaf node. Suppose you are given a set of rules <em>R </em>= {<em>r</em><sub>1</sub><em>,r</em><sub>2</sub><em>,</em>··<em>,r<sub>k</sub></em>}, where <em>r<sub>i </sub></em>corresponds to the <em>i<sup>th </sup></em>rule. Is it possible to convert the rule set <em>R </em>into an equivalent decision tree? Explain your construction or give a counterexample.</li>

 <li>Please read the following paper and briefly summarize the key ideas as you understood (You can skip the proofs, but it is important to understand the main results):</li>

</ol>

Andrew Y. Ng, Michael I. Jordan: On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes. NIPS 2001: 841-848 <a href="http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf">http://ai.stanford.edu/</a><a href="http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf">~</a><a href="http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf">ang/ </a><a href="http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf">papers/nips01-discriminativegenerative.pdf</a>

<ol start="5">

 <li>Naive Bayes vs. Logistic Regression

  <ol>

   <li>Let us assume that the training data satisfies the Naive Bayes assumption (i.e., features areindependent given the class label). As the training data approaches infinity, which classifier will produce better results, Naive Bayes or Logistic Regression? Please explain your reasoning.</li>

   <li>Let us assume that the training data does <strong>NOT </strong>satisfy the Naive Bayes assumption. As the training data approaches infinity, which classifier will produce better results, Naive Bayes or Logistic Regression? Please explain your reasoning.</li>

   <li>Can we compute <em>P</em>(<em>X</em>) from the learned parameters of a Naive Bayes classifier? Please explain your reasoning.</li>

   <li>Can we compute <em>P</em>(<em>X</em>) from the learned parameters of a Logistic Regression classifier? Please explain your reasoning.</li>

  </ol></li>

 <li>Please read the following paper and briefly summarize the key ideas as you understood:</li>

</ol>

Thomas G. Dietterich (1995) Overfitting and under-computing in machine learning. Computing Surveys, 27(3), 326-327. <a href="http://www.cs.orst.edu/~tgd/publications/cs95.ps.gz">http://www.cs.orst.edu/</a><a href="http://www.cs.orst.edu/~tgd/publications/cs95.ps.gz">~</a><a href="http://www.cs.orst.edu/~tgd/publications/cs95.ps.gz">tgd/publications/cs95.ps.gz</a>

<ol start="7">

 <li>We need to perform statistical tests to compare the performance of two learning algorithms on a given learning task. Please read the following paper and briefly summarize the key ideas as you understood:</li>

</ol>

Thomas G. Dietterich: Approximate Statistical Test For Comparing Supervised Classification Learning Algorithms. Neural Computation 10(7): 1895-1923 (1998) <a href="http://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf">http://sci2s.ugr.es/ </a><a href="http://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf">keel/pdf/algorithm/articulo/dietterich1998.pdf</a>

<h1>Programming and Empirical Analysis (5 Percent)</h1>

<ol>

 <li>(<strong>40 points</strong>) Fortune Cookie Classifier<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a></li>

</ol>

You will build a binary fortune cookie classifier. This classifier will be used to classify fortune cookie messages into two classes: messages that predict what will happen in the future (class 1) and messages that just contain a wise saying (class 0). For example,

“Never go in against a Sicilian when death is on the line” would be a message in class 0. “You will get an A in Machine learning class” would be a message in class 1.

<strong>Files Provided </strong>There are three sets of files. All words in these files are lower case and punctuation has been removed.

<ul>

 <li>The training data:</li>

</ul>

traindata.txt: This is the training data consisting of fortune cookie messages. trainlabels.txt: This file contains the class labels for the training data.

<ul>

 <li>The testing data:</li>

</ul>

testdata.txt: This is the testing data consisting of fortune cookie messages. testlabels.txt: This file contains the class labels for the testing data.

<ul>

 <li>A list of stopwords: stoplist.txt</li>

</ul>

There are two steps: the pre-processing step and the classification step. In the pre-processing step, you will convert fortune cookie messages into features to be used by your classifier. You will be using a bag of words representation. The following steps outline the process involved:

Form the vocabulary. The vocabulary consists of the set of all the words that are in the training data with stop words removed (stop words are common, uninformative words such as “a” and “the” that are listed in the file stoplist.txt). The vocabulary will now be the features of your training data. Keep the vocabulary in alphabetical order to help you with debugging.

Now, convert the training data into a set of features. Let M be the size of your vocabulary. For each fortune cookie message, you will convert it into a feature vector of size M. Each slot in that feature vector takes the value of 0 or 1. For these M slots, if the ith slot is 1, it means that the ith word in the vocabulary is present in the fortune cookie message; otherwise, if it is 0, then the ith word is not present in the message. Most of these feature vector slots will be 0. Since you are keeping the vocabulary in alphabetical order, the first feature will be the first word alphabetically in the vocabulary.

Implement the Naive Bayes Classifier (with Laplace Smoothing) and run it on the training data. Compute the training and testing accuracy.

To debug and test your implementation, you can employ Weka (weka.classifiers.bayes.NaiveBayes):

<a href="http://www.cs.waikato.ac.nz/ml/weka/downloading.html">http://www.cs.waikato.ac.nz/ml/weka/downloading.html</a> or scikit-learn (<a href="http://scikit-learn.org/stable/modules/naive_bayes.html">http://scikit</a>-learn. <a href="http://scikit-learn.org/stable/modules/naive_bayes.html">org/stable/modules/naive_bayes.html</a><a href="http://scikit-learn.org/stable/modules/naive_bayes.html">)</a>

<ol start="2">

 <li>(<strong>60 points</strong>) Convolutional Neural Networks (CNNs) for solving image classification task. You will train a CNN on Fashion MNIST data. The network architecture contains 4 CNN layers followed by one pooling layer and a final fully connected layer. The basic architecture (in sequential order) will be as follows:</li>

</ol>

First CNN layer: input channels – 1, output channels – 8, kernel size = 5, padding = 2, stride = 2 followed by ReLU operation

Second CNN layer: input channels – 8, output channels – 16, kernel size = 3, padding = 1, stride = 2 followed by ReLU operation

Third CNN layer: input channels – 16, output channels – 32, kernel size = 3, padding = 1, stride = 2 followed by ReLU operation

Fourth CNN layer: input channels – 32, output channels – 32, kernel size = 3, padding = 1, stride = 2 followed by ReLU operation

one “Average” pooling layer (nn.AdaptiveAvgPool2d(1) would work in PyTorch)

Fully connected layer (nn.Linear in PyTorch) – determine the number of input features from previous CNN layers. This can be done easily by hand. The number of output features will be equal to number of classes, i.e., 10. If you want help, you can use the direct formula given on this page: <a href="https://cs231n.github.io/convolutional-networks/">http://cs231n.github.io/convolutional-networks/</a><a href="https://cs231n.github.io/convolutional-networks/">.</a>

This will be a straightforward extension from the code discussed in the demo session. Plot the training and testing accuracy as a function of atleast 10 epochs. You could use a smaller sized dataset if compute power is a hurdle. A good choice would be 50 percent of the training set and 10 percent of the testing set. Please make sure you have equal ratio of all classes in the dataset. You can try all tips mentioned in the demo session for solving this task. Optionally, it will be a good idea to try adding other training techniques to see the maximum accuracy possible. Some of them include batch normalization, data augmentation, using other optimizers like ADAM etc.

<strong>Instructions for Code Submission and Output Format.</strong>

Please follow the below instructions. It will help us in grading your programming part of the homework. Please submit the zip file on Blackboard.

<ul>

 <li>Supported programming languages: Python, Java, C++</li>

 <li>Store all the relevant files in a folder and submit the corresponding zipfile named after your student-id, e.g., 114513209.zip</li>

 <li>This folder should have a script file named sh</li>

</ul>

Executing this script should do all the necessary steps required for executing the code including compiling, linking, and execution

<ul>

 <li>Assume relative file paths in your code. Some examples:</li>

</ul>

‘‘./filename.txt’’ or ‘‘../hw2/filename.txt’’

<ul>

 <li>The output of your program should be dumped in a file named “output.txt”</li>

 <li>Make sure the output.txt file is dumped when you execute the script sh</li>

 <li>Zip the entire folder and submit it as</li>

</ul>

&lt;student_id&gt;.zip

<h1>Grading Rubric</h1>

Each question in the students work will be assigned a letter grade of either A,B,C,D, or F by the Instructor and TAs. This five-point (discrete) scale is described as follows:

<ul>

 <li><strong>A) Exemplary (=100%)</strong>.</li>

</ul>

Solution presented solves the problem stated correctly and meets all requirements of the problem.

Solution is clearly presented.

Assumptions made are reasonable and are explicitly stated in the solution.

Solution represents an elegant and effective way to solve the problem and is not overly complicated than is necessary.

<ul>

 <li><strong>B) Capable (=75%)</strong>.</li>

</ul>

Solution is mostly correct, satisfying most of the above criteria under the exemplary category, but contains some minor pitfalls, errors/flaws or limitations.

<ul>

 <li><strong>C) Needs Improvement (=50%)</strong>.</li>

</ul>

Solution demonstrates a viable approach toward solving the problem but contains some major pitfalls, errors/flaws or limitations.

<ul>

 <li><strong>D) Unsatisfactory (=25%)</strong></li>

</ul>

Critical elements of the solution are missing or significantly flawed.

Solution does not demonstrate sufficient understanding of the problem and/or any reasonable directions to solve the problem.

<ul>

 <li><strong>F) Not attempted (=0%)</strong></li>

</ul>

No solution provided.

The points on a given homework question will be equal to the percentage assigned (given by the letter grades shown above) multiplied by the maximum number of possible points worth for that question. For example, if a question is worth 6 points and the answer is awarded a <em>B </em>grade, then that implies 4.5 points out of 6.


