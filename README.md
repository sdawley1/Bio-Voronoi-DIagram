# Bio-Voronoi-Diagram

This was the first 'project' I worked on in Python. It was for a summer internship that was not at all related to program or data analysis but I had to make myself useful somehow.

A Voronoi diagram is the partitioning of a region such that for every point located within the region, encapsulating it is the largest possible convex hull. This type of diagram is made possible using a Delaunay triangulation which creates a triangulation for a set of points while maximizing the minimum angle of all the angles in the triangle.

We don't care about the theory behind why it works so much as we care about how the diagram looks. Keeping that idea in mind, apply the same reasoning to the efficiency and time complexity of my code because it certainly is not optimized. 

The ultimate goal of writing this script was to create a program that allowed for biological data to be easily conveyed to audiences with little experience to experts in the field. Based on how Tristan originally presented this data (using Fiji), I think it's safe to say that this program achieved that goal.
