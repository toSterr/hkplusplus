package data;

import algorithms.evolutionary_algorithms.ParameterSet;
import algorithms.problem.BaseIndividual;
import algorithms.problem.TTP;
import basic_hierarchy.implementation.BasicHierarchy;
import basic_hierarchy.interfaces.Hierarchy;
import center.method.CenterMethod;
import distance.measures.Measure;
import distance_measures.Euclidean;
import interfaces.DistanceMeasure;
import interfaces.QualityMeasure;
import internal_measures.*;
import javafx.util.Pair;
import utils.Constans;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class ClustersAndTheirStatistics {
	private Cluster[] clusters;
	private double[][] clustersStDev;
	private double[] clustersAvgVariances;
	private double[] clustersWeights;
	private ArrayList<List<Integer>> clusterChosenNeighbourIndicies; // i-th position is the chosen neighbour cluster ids
	private double statistic;
	private Map<String, QualityMeasure> qualityMeasureNameWithMeasureObject;
	private Map<String, Double> wholeClusteringQualityMeasureNameWithMeasureValue;

	public ClustersAndTheirStatistics(Cluster[] clusters, double statistic, boolean calculateOtherStatistics)
	{
		this.clusters = clusters;
		this.statistic = statistic;
		this.qualityMeasureNameWithMeasureObject = prepareMeasures();

		if(calculateOtherStatistics)
		{
			calculateStandardStatistics();
		}

		this.clusterChosenNeighbourIndicies = new ArrayList<>(clusters.length);
		this.clustersWeights = new double[clusters.length];
		this.wholeClusteringQualityMeasureNameWithMeasureValue = new HashMap<>();

		for(int i = 0; i < clusters.length; i++) {
			this.clusterChosenNeighbourIndicies.add(new ArrayList<>());
		}
	}

	private HashMap<String, QualityMeasure> prepareMeasures() {
		HashMap<String, QualityMeasure> qualityMeasures = new HashMap<>();
		DistanceMeasure measure = new Euclidean();
		//below measures are sensitive to useSubtree toggle
		qualityMeasures.put(FlatCalinskiHarabasz.class.getName(), new FlatCalinskiHarabasz(measure));
		qualityMeasures.put(FlatDaviesBouldin.class.getName(), new FlatDaviesBouldin(measure));
		qualityMeasures.put(FlatDunn1.class.getName(), new FlatDunn1(measure));
		qualityMeasures.put(FlatDunn4.class.getName(), new FlatDunn4(measure));
		qualityMeasures.put(FlatWithinBetweenIndex.class.getName(), new FlatWithinBetweenIndex(measure));
		//above measures are sensitive to useSubtree toggle

		qualityMeasures.put(FlatDunn2.class.getName(), new FlatDunn2(measure));
		qualityMeasures.put(FlatDunn3.class.getName(), new FlatDunn3(measure));

		return qualityMeasures;
	}

	public void calculateInternalMeasures(QualityMeasure clusterWeightQualityMeasure, ParameterSet<Integer, TTP> parameters) {
//		calculateWholeHierarchyMeasures();
		calculateClusterWeights(clusterWeightQualityMeasure, parameters);
	}

	private void calculateClusterWeights(QualityMeasure clusterWeightQualityMeasure, ParameterSet<Integer, TTP> parameters) {
		this.clustersWeights = new double[this.clusters.length];
        List<Pair<Integer, Double>> clusterIndexWithOneObjectiveVal = new ArrayList<>(this.clusters.length);
		int mainObjectiveNumber = parameters.random.nextInt(parameters.evaluator.getNumObjectives());

		for(int i = 0; i < this.getClusters().length; i++) {
			clusterIndexWithOneObjectiveVal.add(new Pair<>(i, this.getClusters()[i].getCenter().getCoordinate(mainObjectiveNumber)));
		}

		clusterIndexWithOneObjectiveVal.sort(new Comparator<Pair<Integer, Double>>() {
            public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
                if (Objects.equals(o1.getValue(), o2.getValue()))
                    return 0;
                return o1.getValue() < o2.getValue() ? -1 : 1;
            }
        });

		for(int i = 0; i < clusterIndexWithOneObjectiveVal.size() && clusterIndexWithOneObjectiveVal.size() > 1; i++) {
			int clusterIndex = clusterIndexWithOneObjectiveVal.get(i).getKey();

			if(i == 0) {
				int neighbourClusterIndex = clusterIndexWithOneObjectiveVal.get(i+1).getKey();
				double clusterWeight =
						calculateClusterWeightBasedOnItsNeighbour(clusterWeightQualityMeasure, clusterIndex, neighbourClusterIndex);
				this.clustersWeights[clusterIndex] = clusterWeight;
				this.clusterChosenNeighbourIndicies.get(clusterIndex).add(neighbourClusterIndex);
			} else if (i == (clusterIndexWithOneObjectiveVal.size() - 1)) { // last cluster
				int neighbourClusterIndex = clusterIndexWithOneObjectiveVal.get(i-1).getKey();
				double clusterWeight =
						calculateClusterWeightBasedOnItsNeighbour(clusterWeightQualityMeasure, clusterIndex, neighbourClusterIndex);
				this.clustersWeights[clusterIndex] = clusterWeight;
				this.clusterChosenNeighbourIndicies.get(clusterIndex).add(neighbourClusterIndex);
			} else {
				//left neighbour
				int leftNeighbourClusterIndex = clusterIndexWithOneObjectiveVal.get(i-1).getKey();
				double leftClusterWeight =
						calculateClusterWeightBasedOnItsNeighbour(clusterWeightQualityMeasure, clusterIndex, leftNeighbourClusterIndex);

				//right neighbour
				int rightNeighbourClusterIndex = clusterIndexWithOneObjectiveVal.get(i-1).getKey();
				double rightClusterWeight =
						calculateClusterWeightBasedOnItsNeighbour(clusterWeightQualityMeasure, clusterIndex, rightNeighbourClusterIndex);

				if(clusterWeightQualityMeasure.isFirstMeasureBetterThanSecond(leftClusterWeight, rightClusterWeight)) {
					this.clustersWeights[clusterIndex] = rightClusterWeight; // we want the algorithm to focus on the worst clustering
					this.clusterChosenNeighbourIndicies.get(clusterIndex).add(rightNeighbourClusterIndex);
				} else {
					this.clustersWeights[clusterIndex] = leftClusterWeight;
					this.clusterChosenNeighbourIndicies.get(clusterIndex).add(leftNeighbourClusterIndex);
				}
			}
		}
	}

	private double calculateClusterWeightBasedOnItsNeighbour(QualityMeasure clusterWeightQualityMeasure,
															 int clusterIndex,
															 int neighbourClusterIndex) {
		Cluster neighbourCluster = this.clusters[neighbourClusterIndex];
		Hierarchy clusterSubsetHierarchy = new BasicHierarchy(new Cluster[]{this.getClusters()[clusterIndex],
																			neighbourCluster});
		double clusterWeight = clusterWeightQualityMeasure.getMeasure(clusterSubsetHierarchy);
		return clusterWeight;
	}

	private void calculateWholeHierarchyMeasures() {
		Hierarchy hierarchyForInternalMeasureComputations = new BasicHierarchy(this.getClusters());
		this.wholeClusteringQualityMeasureNameWithMeasureValue.clear();
		for(Map.Entry<String, QualityMeasure> qm: this.qualityMeasureNameWithMeasureObject.entrySet()) {
			String measureName = qm.getKey();
			QualityMeasure measure = qm.getValue();
			double measureVal = measure.getMeasure(hierarchyForInternalMeasureComputations);
			this.wholeClusteringQualityMeasureNameWithMeasureValue.put(measureName, measureVal);
		}
	}

	private void calculateStandardStatistics() {
		calculateStDevForEachCluster();
		calculateVarianceForEachCluster();
	}

	private void calculateVarianceForEachCluster() { // TODO TRZEBA ZROBIC PRAWDZIWE ODCHYLENIE STANDARDOWE
		clustersAvgVariances = new double[clusters.length];

		for(int i = 0; i < clusters.length; i++)
		{
			clustersAvgVariances[i] = calculateClusterVariance(i);
		}
	}

	private double calculateClusterVariance(int clusterIndex) {
		double[] vars = new double[DataPoint.getNumberOfDimensions()];
		double overallVariance = 0.0;
		for(int k = 0; k < DataPoint.getNumberOfDimensions(); k++)
		{
			for(int j = 0; j < clusters[clusterIndex].getPoints().length; j++)
			{
				double diff = clusters[clusterIndex].getPoints()[j].getCoordinate(k) - clusters[clusterIndex].getCenter().getCoordinate(k);
				vars[k] += diff*diff;
			}
			vars[k] /= (clusters[clusterIndex].getPoints().length);
			overallVariance += vars[k];
		}
		return overallVariance/DataPoint.getNumberOfDimensions();
	}

	private void calculateStDevForEachCluster() { // TODO TRZEBA ZROBIC PRAWDZIWE ODCHYLENIE STANDARDOWE
		clustersStDev = new double[clusters.length][DataPoint.getNumberOfDimensions()];
		
		for(int i = 0; i < clusters.length; i++)
		{
			clustersStDev[i] = sumClusterPoints(i);
			clustersStDev[i] = averageTableValues(clustersStDev[i], clusters[i].getPoints().length);
		}
		
	}

	private double[] sumClusterPoints(int clusterIndex) {
		double[] sum = new double[DataPoint.getNumberOfDimensions()];
		for(int j = 0; j < clusters[clusterIndex].getPoints().length; j++)
		{
			for(int k = 0; k < DataPoint.getNumberOfDimensions(); k++)
			{
				sum[k] += clusters[clusterIndex].getPoints()[j].getCoordinate(k);
			}
		}
		return sum;
	}
	
	private double[] averageTableValues(double[] table, int denominator) {
		for(int i = 0; i < table.length; i++)
		{
			table[i] /= (double)denominator;
		}
		return table;
	}

	public Cluster[] getClusters() {
		return clusters;
	}

	public double getClusterisationStatistic() {
		return statistic;
	}

	public double[] getClustersAvgVariances() {
		return clustersAvgVariances;
	}

	public double[] getClustersWeights() {
		return clustersWeights;
	}

	public ArrayList<List<Integer>> getClusterChosenNeighbourIndicies() {
		return clusterChosenNeighbourIndicies;
	}

	public void toFile(String folderName, String fileName, int minClusterId, int maxClusterId,
					   List<Double> clustersDispersion, List<Double> clusterWeights) {
		try {
			String fullPath = folderName + File.separator + fileName;
			Files.createDirectories(Paths.get(folderName));
			BufferedWriter writer = new BufferedWriter(new FileWriter(fullPath));
			StringBuilder output = new StringBuilder("No of clusters" + Constans.delimiter + this.getClusters().length
					+ Constans.delimiter + "Sum of Inter cluster distances"	+ Constans.delimiter + this.getClusterisationStatistic()
					+ "\n");
			for(Map.Entry<String, Double> qmWithValue: this.wholeClusteringQualityMeasureNameWithMeasureValue.entrySet()) {
				output.append(qmWithValue.getKey() + Constans.delimiter + qmWithValue.getValue() + Constans.delimiter);
			}
			output.append("\n");

			for(int i = 0; i <  this.getClusters().length; i++) {
				Cluster cls = this.getClusters()[i];
				double clsWeight = this.getClustersWeights()[i];
				double modifiedClsWeight = clusterWeights.get(i);
				double clsAvgVar = this.getClustersAvgVariances()[i];
				double modifiedClsAvgVar = clustersDispersion.get(i);
				output.append("centre id" + Constans.delimiter).append(cls.getClusterId()).append("\n");

				if(cls.getClusterId() == minClusterId) {
					output.append("EDGE - MIN CLUSTER" + "\n");
				}

				if(cls.getClusterId() == maxClusterId) {
					output.append("EDGE - MAX CLUSTER" + "\n");
				}
				output.append("Centroid" + Constans.delimiter)
						.append(cls.getCenter()).append("weight").append(Constans.delimiter)
						.append(modifiedClsWeight).append("(" + clsWeight + ")")
						.append(Constans.delimiter).append("avg variance")
						.append(Constans.delimiter).append(modifiedClsAvgVar).append("(" + clsAvgVar + ")")
						.append("\n");
				for(var point: cls.getPoints()) {
					output.append(point).append("\n");
				}
			}
			writer.write(output.toString());
			writer.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}

	public ClustersAndTheirStatistics addGapClusters(Measure measure, CenterMethod centreMethod) {
		if(this.getClusters().length < 2) {
			return this;
		}

		List<Cluster> gapClusters = new ArrayList<>();
		for(int i = 0; i < this.getClusters().length - 1; i++) {
			Cluster firstCluster = this.getClusters()[i];
			DataPoint minIndividual = null;
			double minIndividualVal = Double.MAX_VALUE;
			DataPoint maxIndividual = null;
			double maxIndividualVal = -Double.MIN_VALUE;

			// get the edge individuals
			for(int j = 0; j < firstCluster.getNumberOfPoints(); j++) {
				DataPoint point = firstCluster.getPoints()[j];
				double pointFirstCoordinate = point.getCoordinate(0);
				if(minIndividualVal > pointFirstCoordinate) {
					minIndividualVal = pointFirstCoordinate;
					minIndividual = point;
				}

				if(maxIndividualVal < pointFirstCoordinate) {
					maxIndividualVal = pointFirstCoordinate;
					maxIndividual = point;
				}
			}

			// find the CLOSEST point from ANOTHER cluster to the edge points
			DataPoint secondClusterClosestToMinIndividual = null;
			double secondClusterClosestToMinIndividualVal = Double.MAX_VALUE;
			DataPoint secondClusterClosestToMaxIndividual = null;
			double secondClusterClosestToMaxIndividualVal = Double.MAX_VALUE;
			for(int k = i + 1; k < this.getClusters().length; k++) {
				Cluster secondCluster = this.getClusters()[k];

				for(int l = 0; l < secondCluster.getNumberOfPoints(); l++) {
					DataPoint secondClusterDataPoint = secondCluster.getPoints()[l];
					double secondClusterMinIndDataPointDistance = measure.distance(minIndividual, secondClusterDataPoint);

					if(secondClusterDataPoint.getCoordinate(0) < minIndividualVal && //makes sure we're looking on the proper side of the cluster
							secondClusterMinIndDataPointDistance < secondClusterClosestToMinIndividualVal) {
						secondClusterClosestToMinIndividualVal = secondClusterMinIndDataPointDistance;
						secondClusterClosestToMinIndividual = secondClusterDataPoint;
					}

					double secondClusterMaxIndDataPointDistance = measure.distance(maxIndividual, secondClusterDataPoint);
					if(secondClusterDataPoint.getCoordinate(0) > maxIndividualVal && //makes sure we're looking on the proper side of the cluster
							secondClusterMaxIndDataPointDistance < secondClusterClosestToMaxIndividualVal) {
						secondClusterClosestToMaxIndividualVal = secondClusterMaxIndDataPointDistance;
						secondClusterClosestToMaxIndividual = secondClusterDataPoint;
					}
				}
			}

			if(secondClusterClosestToMinIndividual != null) {
				DataPoint[] minGapClusterPoints = new DataPoint[2];
				minGapClusterPoints[0] = new DataPoint(minIndividual.getCoordinates(),
						minIndividual.getSourceCoordinates(),
						minIndividual.getInstanceName(),
						minIndividual.getClassAttribute());
				minGapClusterPoints[1] = new DataPoint(secondClusterClosestToMinIndividual.getCoordinates(),
						secondClusterClosestToMinIndividual.getSourceCoordinates(),
						secondClusterClosestToMinIndividual.getInstanceName(),
						secondClusterClosestToMinIndividual.getClassAttribute());
				gapClusters.add(centreMethod.makeCluster(minGapClusterPoints, measure));
			}

			if(secondClusterClosestToMaxIndividual != null) {
				DataPoint[] maxGapClusterPoints = new DataPoint[2];
				maxGapClusterPoints[0] = new DataPoint(maxIndividual.getCoordinates(),
						maxIndividual.getSourceCoordinates(),
						maxIndividual.getInstanceName(),
						maxIndividual.getClassAttribute());
				maxGapClusterPoints[1] = new DataPoint(secondClusterClosestToMaxIndividual.getCoordinates(),
						secondClusterClosestToMaxIndividual.getSourceCoordinates(),
						secondClusterClosestToMaxIndividual.getInstanceName(),
						secondClusterClosestToMaxIndividual.getClassAttribute());
				gapClusters.add(centreMethod.makeCluster(maxGapClusterPoints, measure));
			}
		}

		for(int x = 0; x < this.getClusters().length; x++) {
			gapClusters.add(this.getClusters()[x]);
		}
		Cluster[] newClusters = gapClusters.toArray(new Cluster[gapClusters.size()]);
		return new ClustersAndTheirStatistics(newClusters, measure.calculateClusterisationStatistic(newClusters), true);
	}
}
