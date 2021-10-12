# Background


## 1. The Chain Rule
$y = f(g(h(x)))$
From inside to out side: 
$$w1 = h(x)$$
$$w2 = g(w1)$$
$$y = f(w2)$$
The derivative with respect to x: 
$$\frac{dy}{dx} = \frac{dy}{dw_{2}} \frac{dw_{2}}{dw_{1}} \frac{dw_{1}}{dx}$$ 

## 2. Computational Graph
Below is the computational graph of $$f(x_{1}, x_{2}) = \ln(x_{1}) + x_{1}x_{2} - \sin(x_{2})$$


## 3. Elementary Functions
$$\frac{{d}}{{{d}{x}}}{\left({c}\right)}={0}$$
$${\left({{x}}^{{n}}\right)}'={n}{{x}}^{{{n}-{1}}}$$
$${\left({{a}}^{{x}}\right)}'={{a}}^{{x}}{\ln{{\left({a}\right)}}}$$, $${{{{\left({{e}}^{{x}}\right)}'={{e}}^{{x}}}}}$$
$${\log}_{{a}}{\left({x}\right)}=\frac{{1}}{{{x}{\ln{{\left({a}\right)}}}}}$$, $${\left({\ln{{\left({x}\right)}}}\right)}'=\frac{{1}}{{x}}$$
$${\left({\sin{{\left({x}\right)}}}\right)}'={\cos{{\left({x}\right)}}}$$
$${\left({\cos{{\left({x}\right)}}}\right)}'=-{\sin{{\left({x}\right)}}}$$
$${\left({\tan{{\left({x}\right)}}}\right)}'=\frac{{1}}{{{{\cos}}^{{2}}{\left({x}\right)}}}={{\sec}}^{{2}}{\left({x}\right)}$$
$${\left({\operatorname{arcsin}{{\left({x}\right)}}}\right)}'=\frac{{1}}{{\sqrt{{{1}-{{x}}^{{2}}}}}}$$
$${\left({\operatorname{arccos}{{\left({x}\right)}}}\right)}'=-\frac{{1}}{{\sqrt{{{1}-{{x}}^{{2}}}}}}$$
$${\left({\operatorname{arctan}{{\left({x}\right)}}}\right)}'= \frac{{1}}{{{{1}+{{x}}^{{2}}}}}$$


## 4. Forward 

## 5. Reverse 

