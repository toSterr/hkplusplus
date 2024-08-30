package data;

import java.awt.Color;
import java.io.*;

import Jama.Matrix;
import algorithms.Algorithm;
import utils.Constans;

public class Cluster {
	private DataPoint[] points;
	private DataPoint center;//jest urozsamiany z srednia mikstury w gmm
	private Matrix covariance;
	private double mixingCoefficient;
	private boolean staticCenter;
	private static Algorithm algorithm;
	private Color colorOnImage;
	private int parentId;
	private int clusterId;

	public Cluster(DataPoint[] points, DataPoint center, Color clusterColorOnImage, int parentId, int clusterId)
	{
		this.points = points;
		this.center = center;
		this.colorOnImage = clusterColorOnImage;
		this.parentId = parentId;
		this.clusterId = clusterId;
		this.setStaticCenter(false);
	}
	
	public Cluster(DataPoint[] points, DataPoint center, Matrix covariance, double mixingCoefficient, Color clusterColorOnImage, int parentId, int clusterId)
	{
		this(points, center, clusterColorOnImage, parentId, clusterId);
		this.covariance = covariance;
		this.mixingCoefficient = mixingCoefficient;
	}
	
	public int getNumberOfPoints()
	{
		return points.length;
	}

	public DataPoint[] getPoints() {
		return points;
	}

	public void setPoints(DataPoint[] points) {
		this.points = points;
	}

	public DataPoint getCenter() {
		return center;
	}

	public void setCenter(DataPoint center) {
		this.center = center;
	}

	public ClustersAndTheirStatistics performSplit(int k, int numberOfAlreadyCreatedNodes) {
		return algorithm.run(k, this, numberOfAlreadyCreatedNodes);
	}

	public double getMaximumValueOfDataPoints(int dimNumber)
	{
		double returnValue = (-1)*Double.MAX_VALUE;
		for(DataPoint p: points)
		{
			if(p.getCoordinate(dimNumber) > returnValue)
			{
				returnValue = p.getCoordinate(dimNumber); 
			}
		}
		return returnValue;
	}
	
	public double getMinimumValueOfDataPoints(int dimNumber)
	{
		double returnValue = Double.MAX_VALUE;
		for(DataPoint p: points)
		{
			if(p.getCoordinate(dimNumber) < returnValue)
			{
				returnValue = p.getCoordinate(dimNumber); 
			}
		}
		return returnValue;
	}
	
	public static void setAlgorithm(Algorithm algorithm) {
		Cluster.algorithm = algorithm;
	}

	public boolean isStaticCenter() {
		return staticCenter;
	}

	public void setStaticCenter(boolean staticCenter) {
		this.staticCenter = staticCenter;
	}

	public Matrix getCovariance() {
		return covariance;
	}

	public void setCovariance(Matrix covariance) {
		this.covariance = covariance;
	}

	public double getMixingCoefficient() {
		return mixingCoefficient;
	}

	public void setMixingCoefficient(double mixingCoefficient) {
		this.mixingCoefficient = mixingCoefficient;
	}

	public Color getColorOnImage() {
		return colorOnImage;
	}

	public void setColorOnImage(Color colorOnImage) {
		this.colorOnImage = colorOnImage;
	}
	
	public int getParentId() {
		return parentId;
	}

	public int getClusterId() {
		return clusterId;
	}
	
	@Override
	public String toString() {
		String returnValue = "";
		returnValue += "Center: " + (this.center == null ? "null" : this.center);
		returnValue += "\nNum. of Pts: " + (this.points == null ? "null pts" : this.points.length);
		returnValue += "\nMixing coef.: " + this.mixingCoefficient;
		returnValue += "\nStatic: " + (this.staticCenter ? "true" : "false");
		StringWriter sw = new StringWriter();
		PrintWriter pw = new PrintWriter(sw);
		if (this.covariance == null) {
			returnValue += "\nCov. matrix: null";
		} else {
			covariance.print(pw, 7, 4);
			pw.flush();
			pw.close();
			returnValue += "\nCov. matrix: " + sw.toString();
		}
		returnValue += "\nColour on img.: " + (this.colorOnImage == null ? "null" : this.colorOnImage.toString());
		returnValue += "\nParent Id: " + this.parentId;
		returnValue += "\nCluster Id: " + this.clusterId;
		return returnValue;
	}
}
