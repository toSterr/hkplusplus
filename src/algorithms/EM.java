package algorithms;

import utils.Utils;
import Jama.Matrix;
import center.method.CenterMethod;
import data.ClustPtsPostAndSumOfPost;
import data.Cluster;
import data.ClustersAndTheirStatistics;
import data.ColorPalette;
import data.Data;
import data.DataPoint;
import data.MixtureNumberWithItsResponsibility;
import data.Parameters;
import distance.measures.GMMBayesMLE;
import distance.measures.LOG_GMMBayesMLE;
import distance.measures.Measure;

public class EM extends Common implements CenterMethod {
	MixtureNumberWithItsResponsibility[] oldMixturesWithMaxResp;
	MixtureNumberWithItsResponsibility[] newMixturesWithMaxResp;
	int numberOfZeroPosterioriProb;
	
	@Override
	public ClustersAndTheirStatistics run(int k, Cluster parent, int numberOfAlreadyCreatedNodes) 
	{
		Cluster[] clusterisation = null;
		int iterationNumber = 0;
		initialiseMixturesWithMaxResp(parent);
		double[][] pointsToMixturePosteriories = null;
		double[] clustersSumOfPosteriories = null;
		
		if(numberOfAlreadyCreatedNodes + k < Parameters.getMaxNumberOfNodes())
		{
			clusterisation = createInitialMixtures(k, parent);
		}
		else
		{
			int avialableNumberOfNodes = Parameters.getMaxNumberOfNodes() - numberOfAlreadyCreatedNodes;
			if(avialableNumberOfNodes > 0)//TODO: nie zastanawiam sie, czy ma byc static czy nie, wiec moze byc tak, ze dokladnosc liczby nodeow bedzie do +/-1, bo dla avial==1 i statica, stworza sie 2 klastry 
			{
				clusterisation = createInitialMixtures(avialableNumberOfNodes, parent);
			}
			else
			{
				if(clusterisation == null || clusterisation.length == 0)
				{
					clusterisation = new Cluster[1];
					clusterisation[0] = parent;
				}
				return new ClustersAndTheirStatistics(clusterisation, measure.calculateClusterisationStatistic(clusterisation), true);
			}
		}
		
		while(stopCriterionNotMet(iterationNumber))
		{
//			if(Parameters.isVerbose()) {
//				System.out.println("iteration " + iterationNumber);
//			}
			//these two lines below are the elements of Maximization step, but for performance reasons they are here
			pointsToMixturePosteriories = calculateEachPointToEachMixturePosterioriProb(parent, clusterisation);
			clustersSumOfPosteriories = calculateSumOfAllPointsResponsibility(parent, clusterisation, pointsToMixturePosteriories);
			
			ClustPtsPostAndSumOfPost result = removeInsignificantClusters(clusterisation, clustersSumOfPosteriories, pointsToMixturePosteriories);
			clusterisation = result.getClusterisation();
			pointsToMixturePosteriories = result.getPointsToMixturePosteriories();
			clustersSumOfPosteriories = result.getClustersSumOfPosteriories();
			
			clusterisation = expectation(parent, clusterisation, pointsToMixturePosteriories);//TODO przypisanie punktow do centr wykonujemy tylko po to aby porownywac wektory przypisania [new/old]MixturesWithMaxResp
			clusterisation = maximization(parent, clusterisation, pointsToMixturePosteriories, clustersSumOfPosteriories);
			iterationNumber++;
		}
		ClustPtsPostAndSumOfPost resultWithoutEmptyClusters = removeEmptyNonStaticClusters(clusterisation, clustersSumOfPosteriories, pointsToMixturePosteriories);
		clusterisation = resultWithoutEmptyClusters.getClusterisation();
		
		if(Parameters.isClusterReestimationBasedOnItsData())
		{
			clusterisation = clusterReestimationBasedOnItsData(clusterisation);
		}
		
//		System.out.println("=====================================pointsToMixturePosteriories======================================");
//		for(int i = 0; i < pointsToMixturePosteriories.length; i++)
//		{
//			String row = "";
//			double rowSum = 0.0;
//			for(int j = 0; j < pointsToMixturePosteriories[0].length; j++)
//			{
//				row += pointsToMixturePosteriories[i][j] + " ";
//				rowSum += pointsToMixturePosteriories[i][j];
//			}
//			System.out.println(row + "\t\t" + rowSum);
//		}
		
		return new ClustersAndTheirStatistics(clusterisation, measure.calculateClusterisationStatistic(clusterisation), true);
	}

	private Cluster[] clusterReestimationBasedOnItsData(Cluster[] clusterisation) {
		for(int i = 0; i < clusterisation.length; i++)
		{
			if(!clusterisation[i].isStaticCenter())
			{
//				System.out.println(clusterisation[i]);
//				System.out.println("\n");
				clusterisation[i] = recalculateMiuBasedOnClusterData(clusterisation[i]);
				clusterisation[i] = recalculateCovarianceOnClusterData(clusterisation[i]);
//				System.out.println(clusterisation[i]);
//				System.out.println("===========================");
			}
		}
		return clusterisation;
	}

	private Cluster recalculateMiuBasedOnClusterData(Cluster cluster) {
		DataPoint center = calculateAvgPointsCenter(cluster.getPoints());
		cluster.setCenter(center);
		return cluster;
	}

	private Cluster recalculateCovarianceOnClusterData(Cluster cluster) {
		Matrix covariance = calculateSampleCovarianceMatrix(cluster.getPoints(), cluster.getCenter());
		cluster.setCovariance(covariance);
		return cluster;
	}

	private ClustPtsPostAndSumOfPost removeEmptyNonStaticClusters(Cluster[] clusterisation,
			double[] clustersSumOfPosteriories, double[][] pointsToMixturePosteriories) {
		boolean[] toRemove = new boolean[clusterisation.length];
		int newClustersCounter = 0;
		for(int i = 0; i < clusterisation.length; i++)
		{
			if(!clusterisation[i].isStaticCenter() && clusterisation[i].getPoints().length == 0)
			{
				toRemove[i] = true;
				if(Parameters.isVerbose())
				{
					System.err.println("Removing cluster number " + i + " because it's empty.");
				}
			}
			else
			{
				toRemove[i] = false;
				newClustersCounter++;
			}
		}
		
		if(newClustersCounter != clusterisation.length)
		{
			pointsToMixturePosteriories = updatePointsToMixturesPosteriories(pointsToMixturePosteriories, newClustersCounter, toRemove);
			clustersSumOfPosteriories = updateClusterSumOfPosteriories(clustersSumOfPosteriories, newClustersCounter, toRemove);
			clusterisation = getNewClusterisation(clusterisation, toRemove, newClustersCounter);
		}
		
		return new ClustPtsPostAndSumOfPost(clusterisation, pointsToMixturePosteriories, clustersSumOfPosteriories);		
	}

	private double[][] calculateEachPointToEachMixturePosterioriProb(
			Cluster parent, Cluster[] clusterisation) {
		double[][] pointsToMixturesProbabilities = computeConditionalProbOfEachPointToEachMixture(parent, clusterisation);
		double[][] pointsToMixturesPosteriories = new double[parent.getNumberOfPoints()][clusterisation.length];
		numberOfZeroPosterioriProb = 0;
		for(int i = 0; i < parent.getNumberOfPoints(); i++)
		{
			for(int j = 0; j < clusterisation.length; j++)
			{
				if(measure.getClass() == GMMBayesMLE.class)
				{
					pointsToMixturesPosteriories[i][j] = calculateResponsibilityValue(pointsToMixturesProbabilities, clusterisation, i, j);
				}
				else if(measure.getClass() == LOG_GMMBayesMLE.class)
				{
					pointsToMixturesPosteriories[i][j] = calculateLOGResponsibilityValue(pointsToMixturesProbabilities, clusterisation, i, j);
				}
				if(pointsToMixturesPosteriories[i][j] == 0.0d)
				{
					numberOfZeroPosterioriProb++;
				}
			}
		}
//		if(numberOfZeroPosterioriProb > 0)
//		{
//			System.out.println("Number of zero prob: " + numberOfZeroPosterioriProb);
//		}
		return pointsToMixturesPosteriories;
	}
	
	private double[][] computeConditionalProbOfEachPointToEachMixture(// TODO metoda ma duza zlosonosc pamieciowa, ale bedzie sie szybko wykonywala
			Cluster parent, Cluster[] clusterisation) {
		double[][] pointsToMixturesProbabilities = new double[parent.getNumberOfPoints()][clusterisation.length];
		for(int i = 0; i < parent.getNumberOfPoints(); i++)
		{
			for(int j = 0; j < clusterisation.length; j++)
			{
				pointsToMixturesProbabilities[i][j] = measure.distance(clusterisation[j], parent.getPoints()[i]);
			}
		}
		return pointsToMixturesProbabilities;
	}
	
	private double[] calculateSumOfAllPointsResponsibility(// no matter if it is LOG representation or not, the sum is always the sum (without any logs)
			Cluster parent, Cluster[] clusterisation, double[][] pointsToMixturePosteriories) 
	{
		double[] clustersSumOfPosteriories = new double[clusterisation.length];
		for(int i = 0; i < clusterisation.length; i++)
		{
			for(int j = 0; j < parent.getNumberOfPoints(); j++)
			{
				if(measure.getClass() == GMMBayesMLE.class)
				{
					clustersSumOfPosteriories[i] += pointsToMixturePosteriories[j][i];
				}
				else if(measure.getClass() == LOG_GMMBayesMLE.class)
				{
					clustersSumOfPosteriories[i] += Math.exp(pointsToMixturePosteriories[j][i]);
				}
			}
		}
		return clustersSumOfPosteriories;
	}


	private ClustPtsPostAndSumOfPost removeInsignificantClusters(Cluster[] clusterisation,
			double[] clustersSumOfPosteriories, double[][] pointsToMixturePosteriories) {
		boolean[] toRemove = new boolean[clusterisation.length];
		int newClustersCounter = 0;
		
		for(int i = 0; i < clusterisation.length; i++)
		{
			if(!clusterisation[i].isStaticCenter() && Utils.isDoubleExtremelyNearZero(clustersSumOfPosteriories[i]))
			{
				toRemove[i] = true;
				if(Parameters.isVerbose())
				{
					System.err.println("Removing cluster number " + i + " with sum of posteriories: " + clustersSumOfPosteriories[i]);
				}
			}
			else
			{
				toRemove[i] = false;
				newClustersCounter++;
			}
		}
		
		if(newClustersCounter != clusterisation.length)
		{
			pointsToMixturePosteriories = updatePointsToMixturesPosteriories(pointsToMixturePosteriories, newClustersCounter, toRemove);
			clustersSumOfPosteriories = updateClusterSumOfPosteriories(clustersSumOfPosteriories, newClustersCounter, toRemove);
			clusterisation = getNewClusterisation(clusterisation, toRemove, newClustersCounter);
		}
		
		return new ClustPtsPostAndSumOfPost(clusterisation, pointsToMixturePosteriories, clustersSumOfPosteriories);		
	}

	private double[][] updatePointsToMixturesPosteriories(
			double[][] pointsToMixturePosteriories, int newClustersCounter,
			boolean[] toRemove) {
		double[][] updatedPointsToMixturePosteriories = new double[pointsToMixturePosteriories.length][newClustersCounter];
		for(int i = 0; i < updatedPointsToMixturePosteriories.length; i++)
		{
			int newTabCounter = 0;
			for(int j = 0; j < pointsToMixturePosteriories[0].length; j++)
			{
				if(!toRemove[j])
				{
					updatedPointsToMixturePosteriories[i][newTabCounter++] = pointsToMixturePosteriories[i][j];
				}
			}
		}
		return updatedPointsToMixturePosteriories;
	}

	private double[] updateClusterSumOfPosteriories(
			double[] clustersSumOfPosteriories, int newClustersCounter, boolean[] toRemove) 
	{
		double[] updatedClustersSumOfPosteriories = new double[newClustersCounter];
		int newTabCounter = 0;
		for(int i = 0; i < toRemove.length; i++)
		{
			if(!toRemove[i])
			{
				updatedClustersSumOfPosteriories[newTabCounter] = clustersSumOfPosteriories[i];
				newTabCounter++;
			}
		}
		
		return updatedClustersSumOfPosteriories;
	}

	private Cluster[] getNewClusterisation(Cluster[] clusterisation,
			boolean[] toRemove, int newClustersCounter) {
		Cluster[] newClusters = new Cluster[newClustersCounter];
		int nextElementCounter = 0;
		for(int i = 0; i < clusterisation.length; i++)
		{
			if(!toRemove[i])
			{
				newClusters[nextElementCounter++] = clusterisation[i];
			}
		}
		return newClusters;
	}

	private Cluster[] expectation(Cluster parent, Cluster[] clusterisation, double[][] pointsToMixturePosteriories) {
		double maxPosteriori;
		double actualPosterioriProb;
		int bestMixtureIndex = -1;
		int[] mixturesNewSizes = new int[clusterisation.length];
		int[] pointAssignment = new int[parent.getNumberOfPoints()];
		newMixturesWithMaxResp = new MixtureNumberWithItsResponsibility[parent.getNumberOfPoints()];
		for(int i = 0; i < parent.getNumberOfPoints(); i++)
		{
			maxPosteriori = (-1)*Double.MAX_VALUE;
			for(int j = clusterisation.length - 1; j >= 0; --j)//ide od ostatniego elementu, aby (w razie rownosci prawdopodobienstw) obiekt pozostal w static center (on jest ostatnim elementem)
			{
				actualPosterioriProb = pointsToMixturePosteriories[i][j];
				if(actualPosterioriProb > maxPosteriori)
				{
					maxPosteriori = actualPosterioriProb;
					bestMixtureIndex = j;
				}
			}
			newMixturesWithMaxResp[i] = new MixtureNumberWithItsResponsibility(bestMixtureIndex, maxPosteriori); 
			pointAssignment[i] = bestMixtureIndex;
			
			if(bestMixtureIndex ==  -1)
			{
				System.err.println("EM.expectation cannot find pointsToMixturePosteriories for point : " + parent.getPoints()[i] + " which has value!"
						+ " Probabily each pointsToMixturePosteriories[i][j] has value: " + pointsToMixturePosteriories[i][0] + " pobabily it is due to "
						+ "wrong value of E or/and L parameter values. E = " + Parameters.getEpsilon() + " L = " + Parameters.getLittleValue());
				System.exit(1);
			}
			
			mixturesNewSizes[bestMixtureIndex]++;
		}
		clusterisation = movePointsIntoClusters(clusterisation, parent.getPoints(), pointAssignment, mixturesNewSizes);
		return clusterisation;
	}

	private double calculateResponsibilityValue(
			double[][] pointsToMixturesProbabilities,
			Cluster[] clusterisation, int pointNumber, int mixtureNumber) {
		double posterioriNum = clusterisation[mixtureNumber].getMixingCoefficient()*pointsToMixturesProbabilities[pointNumber][mixtureNumber];
		double posterioriDen = 0.0d;
		double returnValue;
		for(int i = 0; i < clusterisation.length; i++)
		{
			posterioriDen += clusterisation[i].getMixingCoefficient()*pointsToMixturesProbabilities[pointNumber][i];
		}
		returnValue = posterioriNum/posterioriDen;
		
		if(Parameters.isStaticCenterResponsibilityScalling() 
				&& clusterisation[mixtureNumber].isStaticCenter())
		{
			returnValue *= Parameters.getResponsibilityScallingFactor();
		}
		
		if(Double.isInfinite(returnValue) || Double.isNaN(returnValue))
		{
			System.err.println("EM.calculateResponsibilityValue returning value is not finite! "
					+ "posterioriNum = clusterisation[mixtureNumber].getMixingCoefficient()*pointsToMixturesProbabilities[pointNumber][mixtureNumber] "
					+ posterioriNum + " = " + clusterisation[mixtureNumber].getMixingCoefficient() + " * " + pointsToMixturesProbabilities[pointNumber][mixtureNumber]
					+ " posterioriDen = " + posterioriDen);
		}
		return returnValue;
	}
	
	private double calculateLOGResponsibilityValue(
			double[][] pointsToMixturesProbabilities,
			Cluster[] clusterisation, int pointNumber, int mixtureNumber) {
		double firstElementValue = Math.log(clusterisation[mixtureNumber].getMixingCoefficient()) + pointsToMixturesProbabilities[pointNumber][mixtureNumber];
		
		double maxOfScaledDistances = (-1)*Double.MAX_VALUE;
		double scaledDistance;
		for(int i = 0; i < clusterisation.length; i++)
		{
			scaledDistance = Math.log(clusterisation[i].getMixingCoefficient()) + pointsToMixturesProbabilities[pointNumber][i];
			if(scaledDistance > maxOfScaledDistances)
			{
				maxOfScaledDistances = scaledDistance;
			}
		}
		
		firstElementValue -= maxOfScaledDistances;
		
		double secondElementSum = 0;
		for(int i = 0; i < clusterisation.length; i++)
		{
			double sumElement = Math.log(clusterisation[i].getMixingCoefficient()) + pointsToMixturesProbabilities[pointNumber][i] - maxOfScaledDistances;
			secondElementSum += Math.exp(sumElement);
		}
		
		double secondElementValue = Math.log(secondElementSum);
		double returnValue = firstElementValue - secondElementValue;
		
		if(Parameters.isStaticCenterResponsibilityScalling() 
				&& clusterisation[mixtureNumber].isStaticCenter())
		{
			returnValue *= Parameters.getResponsibilityScallingFactor();
		}
		
		if(Double.isInfinite(returnValue) || Double.isNaN(returnValue))
		{
			System.err.println("EM.calculateLOGResponsibilityValue returning value is not finite! "
					+ "firstElementValue = Math.log(clusterisation[mixtureNumber].getMixingCoefficient()) + pointsToMixturesProbabilities[pointNumber][mixtureNumber] "
					+ firstElementValue + " = " + Math.log(clusterisation[mixtureNumber].getMixingCoefficient()) + " + " 
					+ pointsToMixturesProbabilities[pointNumber][mixtureNumber]
					+ " secondElementValue = " + secondElementValue);
		}
		return returnValue;
	}
	
	private Cluster[] maximization(Cluster parent, Cluster[] clusterisation, double[][] pointsToMixturePosteriories, double[] clustersSumOfPosteriories) {
		//TODO tak naprawde rekalkulacja odbywa sie wzgledem WSZYTSKICH punktow z parenta, a wiec olewane jest przypisanie punktow do centr
		clusterisation = recalculateMixingCoefficients(parent, clusterisation, clustersSumOfPosteriories);//since clustersSumOfPosteriories are NOT calculated with log, thus we can use this method for both variants without any change
		clusterisation = recalculateMiues(parent, clusterisation, pointsToMixturePosteriories, clustersSumOfPosteriories);//TODO do wyznaczenia nowych miu korzystan ze zrekalkulowanych prawdopod. priori
		clusterisation = recalculateCovarianceMatrices(parent, clusterisation, pointsToMixturePosteriories, clustersSumOfPosteriories);//TODO do rekal;kulacji kowariancji korzystam z nowych miu
		return clusterisation;
	}

	private Cluster[] recalculateMixingCoefficients(Cluster parent, Cluster[] clusterisation, double[] clustersSumOfPosteriories) {
		double newMixingCoefficientValue;
		for(int i = 0; i < clusterisation.length; i++)//uaktualniane jest rowniez mixing coef. dla staticow, bo inaczej beda one sciagaly wszystkie punkty
		{
			newMixingCoefficientValue = clustersSumOfPosteriories[i]/(double)parent.getNumberOfPoints();
			clusterisation[i].setMixingCoefficient(newMixingCoefficientValue);
			
			if(Double.isInfinite(newMixingCoefficientValue) || Double.isNaN(newMixingCoefficientValue))
			{
				System.err.println("EM.recalculateMixingCoefficients, newMixingCoefficientValue is not finite! newMixingCoefficientValue = " + newMixingCoefficientValue
						+ " clustersSumOfPosteriories[i] = " + clustersSumOfPosteriories[i] + " parent.getNumberOfPoints() = " + parent.getNumberOfPoints());
			}
		}
		return clusterisation;
	}
	
	private Cluster[] recalculateMiues(Cluster parent,
			Cluster[] clusterisation, double[][] pointsToMixturePosteriories, double[] clustersSumOfPosteriories) {
		DataPoint newMiu;
		double newMiuCoordinateValue;
		for(int i = 0; i < clusterisation.length; i++)
		{
			if(!clusterisation[i].isStaticCenter())
			{
				newMiu = new DataPoint(new double[DataPoint.getNumberOfDimensions()], null, "miu", "miu");
				for(int j = 0; j < DataPoint.getNumberOfDimensions(); j++)
				{
					newMiuCoordinateValue = 0.0d;
					for(int k = 0; k < parent.getNumberOfPoints(); k++)
					{
						double pointPosteriori = (-1)*Double.MAX_VALUE;
						if(measure.getClass() == GMMBayesMLE.class)
						{
							pointPosteriori = pointsToMixturePosteriories[k][i];
						}
						else if(measure.getClass() == LOG_GMMBayesMLE.class)
						{
							pointPosteriori = Math.exp(pointsToMixturePosteriories[k][i]);
						}
						
						newMiuCoordinateValue += pointPosteriori*parent.getPoints()[k].getCoordinate(j);
					}
					newMiuCoordinateValue /= clustersSumOfPosteriories[i];
					newMiu.setCoordinate(j, newMiuCoordinateValue);
				}
				clusterisation[i].setCenter(newMiu);
			}
		}
		return clusterisation;
	}
	
	private Cluster[] recalculateCovarianceMatrices(Cluster parent,
			Cluster[] clusterisation, double[][] pointsToMixturePosteriories, double[] clustersSumOfPosteriories) {
		double calculationConstant;
		Matrix newCovariance;
		for(int i = 0; i < clusterisation.length; i++)
		{
			if(!clusterisation[i].isStaticCenter())
			{
				calculationConstant = 1.0d/clustersSumOfPosteriories[i];
				Matrix sum = new Matrix(DataPoint.getNumberOfDimensions(), DataPoint.getNumberOfDimensions(), 0.0);
				for(int j = 0; j < parent.getNumberOfPoints(); j++)
				{
					Matrix pointToMeanDifference = parent.getPoints()[j].getMatrix().minus(clusterisation[i].getCenter().getMatrix());
					pointToMeanDifference = pointToMeanDifference.times(pointToMeanDifference.transpose());
					
					if(measure.getClass() == GMMBayesMLE.class)
					{
						pointToMeanDifference = pointToMeanDifference.times(pointsToMixturePosteriories[j][i]);
					}
					else if(measure.getClass() == LOG_GMMBayesMLE.class)
					{
						pointToMeanDifference = pointToMeanDifference.times(Math.exp(pointsToMixturePosteriories[j][i]));
					}
					
					sum = sum.plus(pointToMeanDifference);
					if(Double.isInfinite(sum.get(0, 0)) || Double.isInfinite(sum.get(0, 1)) || Double.isInfinite(sum.get(1, 0)) || Double.isInfinite(sum.get(1, 1))
					   || Double.isNaN(sum.get(0, 0)) || Double.isNaN(sum.get(0, 1)) || Double.isNaN(sum.get(1, 0)) || Double.isNaN(sum.get(1, 1)))
					{
						System.out.println("EM.recalculateCovarianceMatrices() sum matrix have not finite value! Sum matrix values:");
						System.out.println(sum.get(0, 0) + " " + sum.get(0, 1) + "\n" + sum.get(1, 0) + " " + sum.get(1, 1));
						System.out.println("calculationConstant = " + calculationConstant);
						System.out.println("pointToMeanDifference:");
						System.out.println(pointToMeanDifference.get(0, 0) + " " + pointToMeanDifference.get(0, 1) + "\n" + pointToMeanDifference.get(1, 0) + " " 
								+ pointToMeanDifference.get(1, 1));
						System.out.println("parent.getPoints()[j].getMatrix():");
						System.out.println(parent.getPoints()[j].getMatrix().get(0, 0) + " " + parent.getPoints()[j].getMatrix().get(1, 0));
					}
				}
				newCovariance = sum.times(calculationConstant);
				if(Double.isInfinite(newCovariance.get(0, 0)) || Double.isInfinite(newCovariance.get(0, 1)) || Double.isInfinite(newCovariance.get(1, 0)) || Double.isInfinite(newCovariance.get(1, 1))
					|| Double.isNaN(newCovariance.get(0, 0)) || Double.isNaN(newCovariance.get(0, 1)) || Double.isNaN(newCovariance.get(1, 0)) || Double.isNaN(newCovariance.get(1, 1)))
				{
					System.out.println("EM.recalculateCovarianceMatrices() sum matrix multiplied by calculation constant have not finite value! newCovariancenewCovariance matrix values:");
					System.out.println(newCovariance.get(0, 0) + " " + newCovariance.get(0, 1) + "\n" + newCovariance.get(1, 0) + " " + newCovariance.get(1, 1));
				}

				if(Parameters.isDiagonalCovarianceMatrix())
				{
					newCovariance = Utils.getDiagonalMatrix(newCovariance);
				}
				
				clusterisation[i].setCovariance(newCovariance);		
			}
		}
		return clusterisation;
	}

	private Cluster[] createInitialMixtures(int k, Cluster parent) {//https://www.youtube.com/watch?v=BWXd5dOkuTo&feature=player_detailpage#t=628 
		//TODO we could initialise this algorithm by using a k-means method
		DataPoint[] centers = choseSeedPoints(k, parent);//initial centers are estimated directly from data
		Cluster[] clusters;
		
		double priori;
		if(Parameters.isDisableStaticCenter())
		{
			clusters = new Cluster[k];
			priori = (double) (1.0/(double)k);
		}
		else
		{
			clusters = new Cluster[k+1];
			//FIXME this apriori sums up to 1 but SKIPS parent (static center) apriori. Someone could re-think about it some day.
			priori = (double) (1.0/(double)(k+1));//equal aprioris 
		}
		
		for(int i = 0; i < k; i++)
		{
			clusters[i] = new Cluster(null, centers[i], parent.getCovariance().copy(), priori, null, parent.getClusterId(), Utils.getNextId());
		}
		
		if(!Parameters.isDisableStaticCenter())
		{
			if(Parameters.isStaticCenterAdaptiveCovarianve())
			{
				clusters[k] = new Cluster(null, parent.getCenter(), parent.getCovariance().times(1.0/priori), priori, parent.getColorOnImage(),
						parent.getParentId(), parent.getClusterId());// FIXME what appriori assigns here? Maybe there is a better way.
			}
			else if(Parameters.isStaticCenterCovarianceScalling())
			{
				clusters[k] = new Cluster(null, parent.getCenter(), parent.getCovariance().times(1.0/Parameters.getCovarianceScallingFactor()),
						priori, parent.getColorOnImage(), parent.getParentId(), parent.getClusterId());// FIXME what appriori assigns here? Maybe there is a better way.
			}
			else
			{
				clusters[k] = new Cluster(null, parent.getCenter(), parent.getCovariance(), priori, parent.getColorOnImage(), 
						parent.getParentId(), parent.getClusterId());
			}
			clusters[k].setStaticCenter(true);
		}
		return clusters;
	}
	
	private void initialiseMixturesWithMaxResp(Cluster parent) {
		oldMixturesWithMaxResp = new MixtureNumberWithItsResponsibility[parent.getNumberOfPoints()];
		newMixturesWithMaxResp = new MixtureNumberWithItsResponsibility[parent.getNumberOfPoints()];
		
		for(int i = 0; i < oldMixturesWithMaxResp.length; i++)
		{
			oldMixturesWithMaxResp[i] = new MixtureNumberWithItsResponsibility(-1, -1);
			newMixturesWithMaxResp[i] = new MixtureNumberWithItsResponsibility(-1, -1);
		}
		
		if(oldMixturesWithMaxResp.length > 1)//ensure that the tables are different when used for the first time
		{
			oldMixturesWithMaxResp[0] = new MixtureNumberWithItsResponsibility(666, 666);
		}
	}

	private boolean stopCriterionNotMet(int iterationNumber) {
		boolean notMet = true; 
		if(iterationNumber >= Parameters.getNumberOfClusterisationAlgIterations())
		{
			notMet = false;
		}
		if(pointAssignmentNotChanged())
		{
			notMet = false;
		}
		return notMet;
	}

	private boolean pointAssignmentNotChanged() {
		boolean returnValue = true;
		for(int i = 0; i < oldMixturesWithMaxResp.length && returnValue; i++)
		{
			// thanks to that way of comparison, we allows algorithm to better fit the data
			if(newMixturesWithMaxResp[i].getMixtureNumber() != oldMixturesWithMaxResp[i].getMixtureNumber()
					|| newMixturesWithMaxResp[i].getResponsibility() != oldMixturesWithMaxResp[i].getResponsibility())
			{
				returnValue = false;
			}
		}
		oldMixturesWithMaxResp = newMixturesWithMaxResp;//copy of global references
		return returnValue;
	}

	//from interface CenterMethod
	@Override
	public Cluster makeCluster(Data points, Measure measure) {
		DataPoint center = calculateAvgPointsCenter(points.getPoints());
		Matrix covariance = calculateSampleCovarianceMatrix(points.getPoints(), center);
		int rootId = Utils.getNextId();
		Cluster root = new Cluster(points.getPoints(), center, covariance, 1.0d, ColorPalette.getNextColor(), rootId, rootId);	
//		System.out.println(root);
		return root;
	}

	@Override
	public Cluster makeCluster(DataPoint[] points, Measure measure) {
		DataPoint center = calculateAvgPointsCenter(points);
		Matrix covariance = calculateSampleCovarianceMatrix(points, center);
		int rootId = Utils.getNextId();
		Cluster root = new Cluster(points, center, covariance, 1.0d, ColorPalette.getNextColor(), rootId, rootId);
//		System.out.println(root);
		return root;
	}

	private DataPoint calculateAvgPointsCenter(DataPoint[] points) {
		DataPoint center = new DataPoint(new double[DataPoint.getNumberOfDimensions()], null, "miu", "miu");
		double newCoordinateVal;
		for(int i = 0; i < DataPoint.getNumberOfDimensions(); i++)
		{
			newCoordinateVal = 0.0d;
			for(int j = 0; j < points.length; j++)
			{
				newCoordinateVal += points[j].getCoordinate(i);
			}
			newCoordinateVal /= points.length;
			center.setCoordinate(i, newCoordinateVal);
		}
		return center;
	}
	
	private Matrix calculateSampleCovarianceMatrix(DataPoint[] points, DataPoint center) {
		double[][] covarianceMatrixValues = new double[DataPoint.getNumberOfDimensions()][DataPoint.getNumberOfDimensions()];
		
		double covariance;
		for(int i = 0; i < DataPoint.getNumberOfDimensions(); i++)
		{
			for(int j = 0; j < DataPoint.getNumberOfDimensions(); j++)
			{
				covariance = 0.0;
				if(!(Parameters.isDiagonalCovarianceMatrix() && i != j))
				{
					for(int k = 0; k < points.length; k++)
					{
						covariance += (points[k].getCoordinate(i)-center.getCoordinate(i))*
								(points[k].getCoordinate(j)-center.getCoordinate(j));
					}
				}
				//TODO n-1 poniewaz kowariancja probki a nie populacji
				covariance /= (points.length-1);
				covarianceMatrixValues[i][j] = covariance;
			}
		}
		
		return new Matrix(covarianceMatrixValues);
	}

	//from interface CenterMethod
	@Override
	public Cluster updateCenter(Cluster cluster, Measure measure) {
		// it is used only in kmeans
		return null;
	}
	
}
