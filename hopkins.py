import random

def hopkinss(points):
    n=len(points)

    if n==0:
        return None
    
    d=len(points[0])
    m=n//10
    
    mins=[min(point[i] for point in points) for i in range(d)]
    maxs=[max(point[i] for point in points) for i in range(d)]

    U=[]
    for _ in range(m):
        u=[random.uniform(mins[i],maxs[i]) for i in range(d)]
        U.append(u)
    
 
    distance=lambda a,b:sum((a[i]-b[i])**2 for i in range(3))**0.5

    u_sum=0
    for u in U:
        min_distance=float('inf')
        for p in points:
            dist=distance(u,p)
            if dist<min_distance:
                min_distance=dist
        u_sum+=min_distance
    

    indices=random.sample(range(n),m)
    w_sum=0
    for idx in indices:
        w=points[idx]
        min_distance=float('inf')
        for i in range(n):
            if i==idx:
                continue
            x=points[i]
            dist=distance(w,x)
            if dist<min_distance:
                min_distance=dist
        w_sum+=min_distance

    return u_sum/(u_sum+w_sum)



points=[[random.randint(1,100) for _ in range(5)] for _ in range(1000)]

print(hopkinss(points))


