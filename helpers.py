#linear algebra

#we can use the type annotations to define a Vector (which is just an arry of numbers):
from typing import List
Vector = List[float]

height_weight_age = [70, #inches
                     170, #weight
                     40] #age

grades = [95,
          80,
          75,
          62]

#we want to be able to add vectors; However, Lists can't be adding. so to not get an error, we need to define an add functon, antoher thing to consider is that
#if the vectors are not the same size then we can add them.

def add(v: Vector, w: Vector) -> Vector :
    #add the elements of the vectors = v_new = v[i] + w[i]
    assert len(v)==len(w), "vectors must be the same size"
    #we will add each element, and zip them into tuple pairs
    """data=[]
    for num1, num2 in zip(v,w):
        data.append(num1+num2)"""
    #rewrite 
    return [v_i + w_i for v_i, w_i in zip(v,w)]


def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v)==len(w), 'vectors must be the same size'
    return [v_i - w_i for v_i, w_i in zip(v,w)]

v=[1,2,3]
w=[2,3,4]

#print(add(v,w))
#print(subtract(v,w))

#create a test
assert add([1,2,3],[2,3,4])==[3,5,7], "adding vectors isn't working correctly"
assert subtract([1,2,3],[2,3,4])==[-1,-1,-1], "substracting vectors isn't working correctly"

def vector_sum(vectors: List[Vector])->Vector:
    """sums all the corresponding elements"""
    assert vectors, "no vectors provided"
    
    total=[]
    value=0
    num_of_elements = len(vectors[0])
    assert all(len(v) == num_of_elements for v in vectors), "different sizes. no allowed"
    
    """ for j in range(len(vectors[0])):
        for i in range(len(vectors)): 
            value += vectors[i][j]
        total.append(value)
        value=0
    return total"""
    
    #return [ sum(vector[i] for vector in vectors) for i in range(num_of_elements)]

    """so vectors = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]] unpacks to 
           *vectors = [1, 2, 3], [4, 5, 6], [7, 8, 9]
           and then when taking the zip, it will create a series of tuples
           (1,4,7), (2,5,9), (3,6,9) and then summing those values is exactly what we want"""
    return [sum(values) for values in zip(*vectors)] #simplier version. *vector is unpacking the list of vectors to 

#print(f"vector sum:{vector_sum([[1,2],[3,4],[5,6],[7,8]])}")

def scalar_mult(scalar: float,v: Vector) -> Vector:
    return[scalar*v_i for v_i in v]

assert scalar_mult(2,v)==[2,4,6]

#now lets compute the vector element-wise mean. take the sum of the elements and then divide it by number of elements to get the mean
def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_mult(1/n, vector_sum(vectors))

assert vector_mean([[1,2],[3,4],[5,6]]) ==[3,4]

def vector_dot_product(v: Vector, w: Vector) -> float:
    """computes v_1 * w_1 + ... + v_n * w_n"""
    #take their component wise products and then add the results
    #first lets zip the two lists togehter and then iterate across that tuple 
    dp: float = 0
    
    """for v_i, w_i in zip(v,w):
        dp += v_i*w_i
        
    return dp"""
    #rewrite
    return sum(v_i*w_i for v_i, w_i in zip(v,w))

assert vector_dot_product([1,2,3],[4,5,6]) == 32 #1*4 + 2*5 + 3*6

#compute the vectors sum of squares. we cna use the vector_dot_product on itself passing the same vector in v twice.
def sum_of_vector_squares(v: Vector) -> float:
    """compute v_1*v_1 + ... + v_n* v_n"""
    return vector_dot_product(v,v)


#sum_of_vector_squares(v)

import math

#magnitude is the lenght of a given vector.
def vector_magnitude(v: Vector) -> float:
    """computes sqrt(v_1*v_1 + ... + v_n * v_n)"""
    return math.sqrt(sum_of_vector_squares(v))

assert vector_magnitude([3,4]) ==5 
#print(vector_magnitude([3,4]))

#compute the distance between two vectors v,w. formula is sqrt((v_1-w_1)^2 + ... (v_n-w_n)^2)
def vector_distance(v: Vector, w: Vector) -> float:
    #return math.sqrt(sum_of_vector_squares(subtract(v,w)))
    return vector_magnitude(subtract(v,w))

assert vector_distance([1,2],[3,4]) == math.sqrt(8)


#STATISTICS
#used to better understand data. We can use it to distill and communicate relevant features of our data.
def mean(v: Vector) -> float:
    """compute the mean: sum of the values / number of elements"""
    return sum(v_i for v_i in v)/len(v)

#mean(v)

#now we can create a function for the median
def median(v: Vector) -> float:
    """if the number of elements is odd, then take the middle value as the median, 
    otherwise, it is the average of the two middle elements"""
    
    sorted_vector = sorted(v) #we must first sort the list
    num_of_elements = len(v)
    mid_point = (num_of_elements)//2
    if num_of_elements%2==0: #even
        #print(f"even so use these values {sorted_vector[(num_of_elements-1)//2]} and {sorted_vector[(num_of_elements)//2]} ")
        return (sorted_vector[mid_point-1]+ sorted_vector[mid_point])/2
    else:
        return sorted_vector[mid_point]

assert median([1,1,2,2,3,3,4,5,5,6,100])== 3


#the quantile: remember that the median is just the q=0.5 (reprensenting the value under which 50% of the data lies).
def quantile(v: Vector, q: float) -> float:
    """compute the quantile
    1. sort the list
    2. get the q-quantile value"""
    
    return sorted(v)[int(q*len(v))]

#quantile([1,1,2,2,3,3,4,5,5,6,100],0.5)

from collections import Counter
#the mode: the most freqent data point
def mode(v: Vector) -> Vector:
    counts = Counter(v)
    max_count = max(counts.values())
    return [v_i for v_i, count in counts.items() if count==max_count]
    
#mode([1,1,1,2,3,4,5,5,5])

#DISPERSION: the spread of the data

def range_of_data(v: Vector) -> float:
    #get the biggest and smallest elements and then get the difference
   # return sorted(v)[-1]-sorted(v)[0] #could just use the max and min func. probably faster than doing two sorts
    return max(v)-min(v)

#range_of_data(v)

def de_mean(v: Vector) -> Vector:
    x_bar = mean(v) #subtract the mean so that the mean has zero value.
    return [x_i-x_bar for x_i in v]

#variance
def variance_of_data(v: Vector) -> float:
    """The sample variance: almost the average squared deviation/difference from the mean"""
    assert len(v) >=2, "variance requires at least two elements"
    #find the mean
    x_bar=mean(v)
    n = len(v)
    #find the sum of the squared differences between the vector and the value X-bar.
    value = sum((x_i-x_bar)**2 for x_i in v)
    return value/(n-1)

#variance_of_data([1,2,3,4,5])

def standard_deviation_of_data(v: Vector) -> float:
    """the standard deviation is the sqrt of the variance"""
    return math.sqrt(variance_of_data(v))

#standard_deviation_of_data([1,2,3,4,5])

def IQR(v: Vector)->float:
    return quantile(v,0.75) - quantile(v, 0.25)

#IQR([1,2,3,4,5])


def normal_cdf(x: float, mu: float=0, sigma: float=1) -> float:
    return (1 + math.erf((x-mu) / math.sqrt(2)/ sigma )) /2

def inverse_normal_cdf(p: float, mu: float=0, sigma: float=1, tolerance: float =0.0001) -> float:
    """ Find the approximate inverse using binary search """
    #if not standard, compute the standard and rescale
    if mu != 0 or sigma != 1:
       # print("recursion")
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance) # this is essentially making it a standard normal variable

    low_z = -10.0    #normal_cdf(-10) is very close to 0
    high_z = 10.0    #normal_cdf(10) is very close to 1

    while high_z - low_z > tolerance:
        mid_z = (low_z+high_z)/2  #@get the midpoint
        mid_p = normal_cdf(mid_z) 
        #print(f"mid_z = {mid_z}, mid_p = {mid_p}")
        
        if mid_p < p:   
            low_z = mid_z  #midpoint is too low, search above it
        else:
            high_z = mid_z  #midpoint is too hight, search below it

    return mid_z

def covariance(xs: Vector, ys: Vector) -> float:
    return vector_dot_product(de_mean(xs),de_mean(ys))/(len(xs)-1)

        
def correlation(xs: Vector, ys: Vector) -> float:
    """Measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation_of_data(xs)
    stdev_y = standard_deviation_of_data(ys)

    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs,ys) / stdev_x / stdev_y
    else:
        return 0 #if no variation, correlation is zero.


#Matrices
from typing import Tuple, List

#define a Matrix
Matrix = List[List[float]]

def shape(A: Matrix) -> Tuple[int,int]:
    """returns # of rows and columns of a Matrix"""
    rows = len(A)
    columns = len(A[0])
    return (rows, columns)

def get_row(A: Matrix, i: int) -> Vector:
    rows, columns = shape(A)
    assert i <= rows, "you're trying to access a column that doesn't exist"
    return A[i]

def get_column(A: Matrix, i: int) -> Vector:
    rows, columns = shape(A)
    assert i <= columns, "you're trying to access a column that doesn't exist"
    return [A[j][i] for j in range(rows)]


from typing import Callable

#create a Matrix generator functions
def make_a_matrix(num_rows:int, num_col: int, entry_func:Callable[[int,int], float])->Matrix:
    """returns a matrix size num_rows x num_cols, whose entries (i,j)-th entry is entry_func(i,j)"""
    return [[entry_func(i,j)  #given a generator function, create a list or lists.
             for j in range(num_col)] 
            for i in range(num_rows)]


if __name__ == "__main__":

        #define our A, B matrices
    A: Matrix = [[1,2,3],
                 [4,5,6]]
        
    B: Matrix = [[1,2],
                [3,4],
                [5,6]] #3 rows, 2 columns
    
    get_column(A, 0)
    assert shape(A) == (2,3)
    make_a_matrix(4,4,my_func2)
    
    #demo create an identity matrix with 1s on the diagonal.
    make_a_matrix(5,5, lambda i,j: 1 if i==j else 0) #here we will use a lambda function to pass through to create the matrix
    
    #we could also define a matrix null (all zeros)
    def my_func(i:int ,j:int)->float:
        return 1 if i==j else 0
    
    make_a_matrix(6,6,my_func)
    
    #or something more interesting
    def my_func2(i:int, j:int) -> float:
        return i*math.sin(90*j)
