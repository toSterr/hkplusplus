package algorithms;

import java.util.Random;

import data.Cluster;
import data.ClustersAndTheirStatistics;
import data.DataPoint;
import distance.measures.Measure;

public class Common implements Algorithm {
	protected static Measure measure;
	protected static Random randomGenerator = new Random();

	@Override
	public ClustersAndTheirStatistics run(int k, Cluster cls, int numberOfAlreadyCreatedNodes) {
		// TODO Auto-generated method stub
		return null;
	}
	
	protected DataPoint[] choseSeedPoints(int k, Cluster parent) 
	{
		int[] pointsIndices = new int[k];
		DataPoint[] seeds = new DataPoint[k];
		for(int i = 0; i < k; i++)
		{
			do
			{
				pointsIndices[i] = randomGenerator.nextInt(parent.getNumberOfPoints());
				//copy constructor, because these centers will be optimized
				seeds[i] = new DataPoint(parent.getPoints()[pointsIndices[i]].getCoordinates(), null, "centroid (from seed points)", "centroid (from seed points)");
			}
			while(!indicesAreDistinct(pointsIndices, i) || !seedPointsHaveDifferentCoordinates(seeds, i));
		}
		return seeds;
	}

	protected boolean indicesAreDistinct(
			int[] pointsIndices, int currentRandomIndex) 
	{
		for(int i = 0; i < currentRandomIndex; i++)
		{
			if(pointsIndices[i] == pointsIndices[currentRandomIndex])
			{
				return false;
			}
		}
		return true;
	}
	
	private boolean seedPointsHaveDifferentCoordinates(DataPoint[] seeds, int indexOfRecentlyChoosenSeedPoint) {
		for(int i = 0; i < indexOfRecentlyChoosenSeedPoint; i++)
		{
			if(seeds[indexOfRecentlyChoosenSeedPoint].arePointsPointingTheSame(seeds[i]))
			{
				return false;
			}
		}
		return true;
	}
	
	protected Cluster[] movePointsIntoClusters(Cluster[] newClusters,//TODO sprawdzic profilerem, czy to za dlugo nie trwa, rozwazyc, zeby kazdy obiekt Cluster posiadal ArrayListe punktow, wtedy szybciej bedzie sie dodwac
			DataPoint[] points, int[] pointsAssignment, int[] newClustersSizes) 
	{
		DataPoint[][] eachClusterDataPoints = prepareEachClusterPoints(newClusters, newClustersSizes);
		eachClusterDataPoints = fillEachClusterPointsStructure(newClusters, points, pointsAssignment,
				eachClusterDataPoints);
		newClusters = setEachClusterPoints(newClusters, eachClusterDataPoints);
		
		return newClusters;
	}
	
	protected DataPoint[][] prepareEachClusterPoints(Cluster[] newClusters,
			int[] newClustersSizes) 
	{
		DataPoint[][] eachClusterDataPoints = new DataPoint[newClusters.length][];
		for(int i = 0; i < newClusters.length; i++)
		{
			eachClusterDataPoints[i] = new DataPoint[newClustersSizes[i]];
		}
		return eachClusterDataPoints;
	}
	
	protected DataPoint[][] fillEachClusterPointsStructure(Cluster[] newClusters,
			DataPoint[] points, int[] pointsAssignment,
			DataPoint[][] eachClusterDataPoints) 
	{
		int[] eachClusterPointsCounter = new int[newClusters.length];
		int currentCluster;
		int currentClusterPointsCounter;
		for(int i = 0; i < pointsAssignment.length; i++)
		{
			currentCluster = pointsAssignment[i];
			currentClusterPointsCounter = eachClusterPointsCounter[currentCluster];
			eachClusterDataPoints[currentCluster][currentClusterPointsCounter] = points[i];
			eachClusterPointsCounter[currentCluster]++;
		}
		return eachClusterDataPoints;
	}

	protected Cluster[] setEachClusterPoints(Cluster[] newClusters,
			DataPoint[][] eachClusterDataPoints) 
	{
		for(int i = 0; i < newClusters.length; i++)
		{
			newClusters[i].setPoints(eachClusterDataPoints[i]);
		}
		return newClusters;
	}

	public static void setMeasure(Measure measure) {
		Common.measure = measure;
	}

	public static Measure getMeasure() {
		return measure;
	}
}
