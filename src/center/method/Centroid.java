package center.method;

import utils.Utils;
import data.Cluster;
import data.ColorPalette;
import data.Data;
import data.DataPoint;
import distance.measures.Measure;

public class Centroid implements CenterMethod{

	@Override
	public Cluster makeCluster(Data points, Measure measure)
	{
		double[] centroid = measure.updateCenter(points.getPoints());
		int rootId = Utils.getNextId();
		return new Cluster(points.getPoints(), new DataPoint(centroid, null, "centroid", "centroid"), ColorPalette.getNextColor(), rootId, rootId);
	}

	@Override
	public Cluster makeCluster(DataPoint[] points, Measure measure)
	{
		double[] centroid = measure.updateCenter(points);
		int rootId = Utils.getNextId();
		return new Cluster(points, new DataPoint(centroid, null, "centroid", "centroid"), ColorPalette.getNextColor(), rootId, rootId);
	}

	@Override
	public Cluster updateCenter(Cluster cluster, Measure measure) {
		double[] newCoordinates = measure.updateCenter(cluster.getPoints());
		cluster.setCenter(new DataPoint(newCoordinates, null, cluster.getCenter().getInstanceName(),cluster.getCenter().getClassAttribute()));
		return cluster;
	}
}
