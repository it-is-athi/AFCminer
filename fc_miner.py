from collections import defaultdict
from FCA import ConceptBuilder
#the fairness check is done as given in the paper except using B we used the modified adjacency matrix
#because B is defined differently while explaining and AFCMiner algorithm and here as different 
#so in program we cant use the same variable for different casses so instead of that we uses modified adjacency matrix
def FairnessFilter(X1,X2,attributes,C):
    if (len(X1)*len(X2))%len(attributes)!=0:return False
    ANumber=[]
    for a in attributes:
        count=0
        for v in X1:
            count+=C[v][a]
        ANumber.append(count)
    Num=ANumber[0]
    for i in range(1,len(ANumber)):
        if Num!=ANumber[i]:
            return False
    return True
#This code is to filter the cliques based on the parameter K and delta so that it filters weak fair clique 
#instead of absolute fair cliquers
def WFC_FairnessFilter(X,attributes,C,k=1,delta=1):
    counts = [sum(C[v][a] for v in X) for a in attributes]
    if not counts:
        return False
    min_cnt=min(counts)
    max_cnt=max(counts)
    if min_cnt<k or (max_cnt-min_cnt)>delta:
        return False
    return True
#the below fucntion is used to generate powerset as given in the paper
def AttributedConceptsDerivation(X):
    res=[set()]
    for v in X:
        temp=[]
        for sub in res:
            temp.append(sub|{v})
        res+=temp
    return res
#the paramter node_attribute_set is node union attribute set mentioned in the paper
def FairCliqueMiner(V, node_attribute_set, R, k=1, delta=1):
    absolute_fair_cliques = []
    weak_fair_cliques = []
    largest_absolute_fair_size = 0
    largest_weak_fair_size = 0
    attributes=set(node_attribute_set)-set(V)
    absolute_fair_count_by_size = defaultdict(int)
    weak_fair_count_by_size = defaultdict(int)
    #creating the incidence matrix
    Matrix=defaultdict(lambda:defaultdict(int))
    #creating self loop
    for v in V:
        Matrix[v][v] = 1
    for i,j in R:
        Matrix[i][j]=1
        if j in V:
            Matrix[j][i]=1 
    concepts=ConceptBuilder(Matrix,V,node_attribute_set)
    for X1,X2,B in concepts:
        if X1==X2:
            if FairnessFilter(X1,X2,attributes,Matrix):
                absolute_fair_cliques.append(X1)
                absolute_fair_count_by_size[len(X1)] += 1
                largest_absolute_fair_size = max(
                    len(X1),
                    largest_absolute_fair_size
                )
            else:
                powerset=AttributedConceptsDerivation(X1)
                #the powerset is sorted to avoid non maximal subclique 
                #for example if 1,2,3,4 is fair then 1,2 and 3,4 even if its fair its not maximal
                powerset.sort(key=lambda x:len(x),reverse=True)
                #the below part is not same as the paper as the paper is so much generalized in this part
                #the paper doesnt mention how to get the maximal clique alone as mentioned in the above comment
                #the below implementation is to avoid taking cliques from already found fair maximal clique
                cur_maxi=[]
                for sub in powerset:
                    if not sub:continue
                    #the flag is used so that if the current subset is an subset of already found maximal fair clique
                    flag=False
                    for cur in cur_maxi:
                        if (cur&sub)==sub:
                            flag=True
                            break
                    if flag:continue
                    if FairnessFilter(sub,sub,attributes,Matrix):
                        cur_maxi.append(sub)
                        absolute_fair_cliques.append(sub)
                        absolute_fair_count_by_size[len(sub)] += 1
                        largest_absolute_fair_size = max(
                            len(sub),
                            largest_absolute_fair_size
                        )
            
            if WFC_FairnessFilter(X1, attributes, Matrix, k=k, delta=delta):
                weak_fair_cliques.append(X1)
                weak_fair_count_by_size[len(X1)] += 1
                largest_weak_fair_size = max(
                    len(X1),
                    largest_weak_fair_size
                )
            else:
                powerset = AttributedConceptsDerivation(X1) if 'powerset' not in locals() else powerset
                powerset.sort(key=lambda x: len(x), reverse=True)
                cur_maxi=[]
                for sub in powerset:
                    if not sub:
                        continue
                    flag=False
                    for cur in cur_maxi:
                        if (cur&sub)==sub:
                            flag=True
                            break
                    if flag:continue
                    if WFC_FairnessFilter(sub, attributes, Matrix, k=k, delta=delta):
                        cur_maxi.append(sub)
                        weak_fair_cliques.append(sub)
                        weak_fair_count_by_size[len(sub)] += 1
                        largest_weak_fair_size = max(
                            len(sub),
                            largest_weak_fair_size
                        )

    print(
        len(absolute_fair_cliques),
        len(weak_fair_cliques)
    )
    print(
        largest_absolute_fair_size,
        largest_weak_fair_size
    )

    return (
        absolute_fair_cliques,
        weak_fair_cliques,
        largest_absolute_fair_size,
        largest_weak_fair_size,
        absolute_fair_count_by_size,
        weak_fair_count_by_size
    )
