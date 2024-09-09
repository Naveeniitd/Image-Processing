<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
<head>
  
 


   
</head>
<body>
<header id="title-block-header">
<h1 class="title">COL783 Assignment 1</h1>
<p class="subtitle">Due date: 29 August 2024</p>
</header>
<p>This assignment deals with basic image operations, intensity
transformations, spatial filtering, and colour image processing.</p>
<p>The assignment is in two parts. Part 1 is being released on 8 August.
Part 2 will be released once we have covered spatial filtering and
colour image processing.</p>
<section id="part-1" class="level2">
<h2>Part 1</h2>
<ol type="1">
<li><p>You are given a small 100×100 image of the IITD CSE department’s
logo: <a href="cse-logo.png">cse-logo.png</a>. Take another <span
class="math inline">\(M\times N\)</span> image, for example a photograph
you have taken yourself (resize it using any tool, e.g. <a
href="https://www.gimp.org/">GIMP</a>, so that it is of reasonable size,
<span class="math inline">\(1000\le\max(M,N)\le1500\)</span>). Your task
is to rescale the logo so its height and width are <span
class="math inline">\(0.25\max(M,N)\)</span>, and then draw it in the
bottom-left corner of the other image, making the white background of
the logo transparent.</p>
<ol type="1">
<li><p>Let the logo image be <span class="math inline">\(f\)</span>. To
determine the background pixels, find the most common intensity <span
class="math inline">\(r^*\)</span>, and create a new image <span
class="math inline">\(\alpha\)</span> in which <span
class="math inline">\(\alpha(x,y)=1\)</span> if <span
class="math inline">\(|f(x,y)-r^*|\le t\)</span> and 0 otherwise. Choose
a tolerance <span class="math inline">\(t\ge0\)</span> which gives a
reasonable separation of the foreground and background region. Report
both <span class="math inline">\(r^*\)</span> and <span
class="math inline">\(t\)</span>.</p></li>
<li><p>Implement image rescaling using two interpolation algorithms:
nearest neighbour and bilinear interpolation. Perform rescaling of both
<span class="math inline">\(f\)</span> and <span
class="math inline">\(\alpha\)</span> to the target size mentioned
above, and show the results for both algorithms.</p></li>
<li><p>Drawing the logo <span class="math inline">\(f\)</span> on top of
<span class="math inline">\(g\)</span> amounts to taking a weighted
average of the corresponding pixels, <span class="math inline">\(\alpha
f+(1-\alpha)g\)</span>, so that the logo shows up where <span
class="math inline">\(\alpha=1\)</span> and the other image shows where
<span class="math inline">\(\alpha=0\)</span>. Note that this operation,
taken literally, can only be applied when the two images are the same
size, but here <span class="math inline">\(f\)</span> is smaller even
after rescaling. Figure out what to do in order to place the rescaled
<span class="math inline">\(f\)</span> at the bottom-left corner of
<span class="math inline">\(g\)</span>, and explain it in your
report.</p></li>
</ol></li>
<li><p>Choose one of the following high-dynamic-range images to work
with: <a href="groveC.hdr">groveC.hdr</a>, <a
href="groveD.hdr">groveD.hdr</a>, <a
href="memorial.hdr">memorial.hdr</a>, <a href="nave.hdr">nave.hdr</a>,
<a href="vinesunset.hdr">vinesunset.hdr</a>. (These were <a
href="https://www.pauldebevec.com/Research/HDR/">acquired by Paul
Debevec in 1997</a>.) The HDR file format is supported by most image I/O
libraries. Loading it should give you an <span
class="math inline">\(M\times N\times3\)</span> array of floating-point
numbers; convert it to a 1-component <span class="math inline">\(M\times
N\)</span> image by averaging the 3 components of each pixel.</p>
<ol type="1">
<li><p>Report the maximum and minimum intensities in the image, and the
contrast ratio <span class="math inline">\(r_{\max}/r_{\min}\)</span>.
Let us verify that it is difficult to adequately display this range
simply by linearly scaling the intensities, <span
class="math inline">\(s=cr\)</span> for some constant <span
class="math inline">\(c\)</span>. Produce one image in which <span
class="math inline">\(c\)</span> is chosen to map <span
class="math inline">\(r_{\max}\)</span> to 255, and another in which
<span class="math inline">\(c\)</span> maps <span
class="math inline">\(r_{\min}\)</span> to 1; in both cases, remember to
clip the computed intensities to the valid range [0, 255].</p></li>
<li><p>One way to visualize a very large range is to perform a log
transformation, <span class="math inline">\(s = a\log r+b\)</span>.
Choose <span class="math inline">\(a,b\)</span> so that the output
intensities span [0, 255], and display the resulting image.</p></li>
<li><p>Suppose we take logs, apply a linear transformation, and then
undo the log: <span class="math inline">\(s = \exp(a\log(r)+b)\)</span>.
Choose <span class="math inline">\(a,b\)</span> to map the intensity
range <span class="math inline">\([r_{\min},r_{\max}]\)</span> to [1,
255]. In your report, show that this is equivalent to a gamma
transformation.</p></li>
<li><p>Implement histogram equalization and test whether it works well
to display the HDR image. Since the HDR image may have a very large
number of intensity levels spanning many orders of magnitude, I suggest
applying a log transformation, then choosing 256 histogram buckets
spanning the range <span class="math inline">\([\log r_{\min},\log
r_{\max}]\)</span>. In your report, explain whether the log
transformation should significantly affect the results of histogram
equalization or not (apart from quantization issues).</p></li>
<li><p>Now that you have histogram equalization, you can also implement
histogram matching. Demonstrate this by matching the HDR image’s
histogram to that of a real photograph of your choice (which need not be
taken by you).</p></li>
</ol></li>
</ol>
</section>
<section id="part-2" class="level2">
<h2>Part 2</h2>
<ol start="3" type="1">
<li><p>To better preserve local detail, we can try using spatial
filtering to reduce only the contrast of large-scale intensity
variations. In this part, let us work in the “log domain”, similar to
problem 2(c): apply a log transformation, perform whatever spatial
filtering operations, and in the end apply the inverse of the log
transformation (i.e. exp) to the final image.</p>
<ol type="1">
<li><p>Implement Gaussian filtering with a user-specified <span
class="math inline">\(\sigma\)</span>. Use a separable kernel so that
the time complexity is linear in <span
class="math inline">\(\sigma\)</span> rather than quadratic. Demonstrate
the results on an image of your choice, with two significantly different
choices of <span class="math inline">\(\sigma\)</span>.</p></li>
<li><p>Take the log-HDR image, <span class="math inline">\(\hat f = \log
f\)</span>, and decompose it into a lowpass-filtered image <span
class="math inline">\(\hat g\)</span> and a highpass-filtered image
<span class="math inline">\(\hat h\)</span> using your Gaussian filter.
Apply your choice of contrast reduction from problem 2 to only <span
class="math inline">\(\hat g\)</span>, recompose the image by adding the
lowpass and highpass images, and finally undo the log. Show all
intermediate images (with intensities of each suitably rescaled to the
displayable range) as well as the final result.</p></li>
<li><p>Implement a bilateral filter with two user-specified parameters
<span class="math inline">\(\sigma_s\)</span> and <span
class="math inline">\(\sigma_r\)</span>. Apply it to the same image as
in part (a), with parameters chosen to clearly demonstrate that
low-contrast variations are smoothed while high-contrast edges are
preserved.</p></li>
<li><p>Repeat part (b) using a bilateral filter instead of a Gaussian
filter. Again, try to choose parameters to get the best image you can,
in terms of enhancing local details while avoiding artifacts such as
saturation and halos.</p></li>
</ol></li>
<li><p>Now, we will process the original RGB values of the HDR image. We
can use the HSI colour model to apply contrast reduction to the
intensity component while preserving the original chromaticity.</p>
<ol type="1">
<li><p>Visualize the structure of the HSI model by creating an image
where hue is constant, saturation increases horizontally from 0 to 1,
and intensity increases vertically from 0 to 1. Highlight colours which
are outside the standard RGB gamut (i.e. any of R, G, or B is less than
0 or greater than 1) by assigning them a different “error” colour.
Similarly, create one image where saturation is constant, and one where
intensity is constant.</p></li>
<li><p>Show that the HSI formulas in the textbook are meaningful even if
we have HDR colours where R, G, B are arbitrarily large. In particular,
show that if a colour <span class="math inline">\(\mathbf
c=(r,g,b)\)</span> has HSI components <span
class="math inline">\((h,s,i)\)</span>, then for any multiple <span
class="math inline">\(\mathbf c&#39;=(kr,kg,kb)\)</span> we have <span
class="math inline">\(h&#39;=h\)</span>, <span
class="math inline">\(s&#39;=s\)</span>, and <span
class="math inline">\(i&#39;=ki\)</span> no matter how large <span
class="math inline">\(k\)</span> is.</p></li>
<li><p>Process the colour HDR image by converting to HSI, applying
contrast reduction from problem 3(d) to the I component, and converting
back to RGB. Compare the results with applying contrast reduction to the
R, G, and B components independently.</p></li>
<li><p>The values in the original RGB image are in a “linear” colour
space (i.e. proportional to the actual physical intensity of light), but
your display probably expects gamma-corrected RGB values. Apply a gamma
transformation of <span class="math inline">\(\gamma=1/2.2\)</span>,
corresponding to the typical sRGB gamma, to the results of part (c), and
display the images again. In your report, comment on the change in
chromaticity and explain why it occurs.</p></li>
</ol></li>
</ol>
</section>
<section id="submission" class="level2">
<h2>Submission</h2>
<p>Submit a zip file that contains (i) all your code for the assignment,
and (ii) a PDF report that includes your results for each of the
assignment problems. In the report, each output image must be
accompanied with a brief description of the procedure used to create it,
and the values of any relevant parameters.</p>
<p>All images should ideally be saved in PNG format, which is lossless
and so does not cause any information loss (other than quantization if
the intensities are not already in 8-bit format). JPEG is permitted if
your submission PDF is becoming too big for the upload limit.</p>
<p>Your assignment should be submitted on Moodle before midnight on the
due date. Only one person in a group needs to submit. Late days are
counted with a quantization of 0.5 days: if you cannot finish the
assignment by midnight, get some sleep and submit by noon the following
day.</p>
<p>Separately, each of you will individually submit a short response in
a separate form regarding how much you and your partner contributed to
the work in this assignment.</p>
</section>
</body>
</html>
