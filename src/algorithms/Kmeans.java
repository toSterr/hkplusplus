package algorithms;

import utils.Utils;
import center.method.CenterMethod;
import data.Cluster;
import data.ClustersAndTheirStatistics;
import data.DataPoint;
import data.Parameters;

import java.util.ArrayList;

//TODO: oigarnac kmeans dla flagi wylaczajacej static center!  

public class Kmeans extends Common
{
	private static CenterMethod centerMethod;
	private int[] oldPointAssignment;
	private int[] newPointAssignment;	
	
	@Override
	public ClustersAndTheirStatistics run(int k, Cluster parent, int doAlignu)//TODO: wyalignowac sprawe z limitem nodeow dla kmeansa
	{
		Cluster[] newClusters = createInitialClusters(k, parent);
		int iterationNumber = 0;
		initialisePointAssignmentTables(parent);
		while(stopCriterionNotMet(iterationNumber))
		{
			newClusters = assignPointsToClusters(parent, newClusters);
			newClusters = updateClusterCenters(newClusters);
			iterationNumber++;
		}
		newClusters = removeEmptyClusters(newClusters);
		return new ClustersAndTheirStatistics(newClusters, measure.calculateClusterisationStatistic(newClusters), true);
	}

	private Cluster[] removeEmptyClusters(Cluster[] currentClusters) {
		ArrayList<Cluster> newClusters = new ArrayList<>();
		for(var cluster: currentClusters) {
			if(cluster.getPoints().length > 0) {
				newClusters.add(cluster);
			}
		}
		return newClusters.toArray(new Cluster[newClusters.size()]);
	}

	private void initialisePointAssignmentTables(Cluster parent) {
		oldPointAssignment = new int[parent.getNumberOfPoints()];
		newPointAssignment = new int[parent.getNumberOfPoints()];
		if(oldPointAssignment.length > 0)//zapewniam, ze tablice na starcie alg. beda ROZNE
		{
			oldPointAssignment[0] = -1;
		}
	}

	private Cluster[] assignPointsToClusters(Cluster parent,
			Cluster[] clusterisation) 
	{
		double minDistance;
		double actualDistance;
		int minClusterIndex = -1;
		newPointAssignment = new int[parent.getNumberOfPoints()];
		int[] clustersNewSizes = new int[clusterisation.length];
		for(int i = 0; i < parent.getNumberOfPoints(); i++)
		{
			minDistance = Double.MAX_VALUE;
			for(int j = 0; j < clusterisation.length; j++)
			{
				actualDistance = measure.distance(clusterisation[j], parent.getPoints()[i]);
				if(actualDistance < minDistance)
				{
					minDistance = actualDistance;
					minClusterIndex = j;
				}
			}
			newPointAssignment[i] = minClusterIndex;
			clustersNewSizes[minClusterIndex]++;
		}
		clusterisation = movePointsIntoClusters(clusterisation, parent.getPoints(), newPointAssignment, clustersNewSizes);
		return clusterisation;
	}
	
	private Cluster[] updateClusterCenters(Cluster[] newClusters) 
	{
		for(int i = 0; i < newClusters.length; i++)//pomijam ostatni klaster, ktory sie nie rusza
		{
			newClusters[i] = centerMethod.updateCenter(newClusters[i], measure);
		}
		return newClusters;
	}

	private Cluster[] createInitialClusters(int k, Cluster parent) 
	{
		DataPoint[] centers = choseSeedPoints(k, parent); //TODO a jakby dystrybulowac seedy NAOKOLO static centra? Na okregu o promieniu jakimstam, tak aby otaczaly one nasz static point?)
		Cluster[] clusters = new Cluster[k];
		for(int i = 0; i < k; i++)
		{
			clusters[i] = new Cluster(null, centers[i], null, parent.getClusterId(), Utils.getNextId());
		}
		return clusters;
	}
	
	private boolean stopCriterionNotMet(int iterationNumber) 
	{
		boolean notMet = true; 
		if(iterationNumber >= Parameters.getNumberOfClusterisationAlgIterations())
		{
			notMet = false;
		}
		if(pointAssignmentNotChanged())
		{
			notMet = false;
		}
		//TODO wprowadzic kryterium mowiace o tym, ze centra klastrow sie juz NIE RUSZAJA
		//TODO albo sprobowac wyrazic PROCENT poprzedniego ruchu klastrow. Jezeli obcene przemieszczenie klastrow stanowi okolo 10% poprzedniego przemieszczenia, to zakonczyc dzialanie 
		return notMet;
	}

	private boolean pointAssignmentNotChanged() {
		boolean returnValue = true;
		for(int i = 0; i < oldPointAssignment.length && returnValue; i++)
		{
			if(newPointAssignment[i] != oldPointAssignment[i])
			{
				returnValue = false;
			}
		}
		oldPointAssignment = newPointAssignment;//kopiowanie globalnych referencji
		return returnValue;
	}

	public static void setCenterMethod(CenterMethod centerMethod) {
		Kmeans.centerMethod = centerMethod;
	}
}
