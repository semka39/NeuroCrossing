using System;
using System.Collections.Generic;
using System.Linq;

public class DBSCAN
{
    private readonly double _eps;
    private readonly int _minPts;
    private int _clusterId = -1;

    public DBSCAN(double eps, int minPts)
    {
        _eps = eps;
        _minPts = minPts;
    }

    public int[] Fit(double[][] data)
    {
        int[] labels = Enumerable.Repeat(-2, data.Length).ToArray();

        for (int i = 0; i < data.Length; i++)
        {
            if (labels[i] != -2) continue; // Уже посещена

            var neighbors = GetNeighbors(data, i);
            if (neighbors.Count < _minPts)
            {
                labels[i] = -1; // Шум
                continue;
            }

            _clusterId++;
            ExpandCluster(data, labels, i, neighbors);
        }
        return labels;
    }

    private void ExpandCluster(double[][] data, int[] labels, int point, List<int> neighbors)
    {
        labels[point] = _clusterId;
        var queue = new Queue<int>(neighbors);

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            if (labels[current] == -1) labels[current] = _clusterId;
            if (labels[current] != -2) continue;

            labels[current] = _clusterId;
            var currentNeighbors = GetNeighbors(data, current);
            if (currentNeighbors.Count >= _minPts)
            {
                foreach (var n in currentNeighbors)
                {
                    if (labels[n] == -2) queue.Enqueue(n);
                }
            }
        }
    }

    private List<int> GetNeighbors(double[][] data, int index)
    {
        var neighbors = new List<int>();
        for (int i = 0; i < data.Length; i++)
        {
            if (i != index && Distance(data[index], data[i]) <= _eps)
                neighbors.Add(i);
        }
        return neighbors;
    }

    private double Distance(double[] p1, double[] p2)
    {
        return Math.Sqrt(p1.Zip(p2, (a, b) => (a - b) * (a - b)).Sum());
    }
}
