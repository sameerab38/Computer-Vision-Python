/**
 * Author: sbammidi
 * Date created: 05/06/2018
 */


import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import javax.imageio.ImageIO;

import org.apache.commons.io.FileUtils;

public class DeepLearningInputConverter {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		File indir = new File("/home/sameera/yolo_exp/darknet/ProjectDataSet/test_dataset/test_data/images/oldtxts");
		
		File imgdir = new File("/home/sameera/yolo_exp/darknet/ProjectDataSet/test_dataset/test_data/images");
		//String[] imgs = imgdir.list( new SuffixFileFilter(".jpg") );
		
		
		PrintWriter pwr = new PrintWriter(new FileWriter("/home/sameera/yolo_exp/darknet/ProjectDataSet/test_labels.csv"));
		pwr.println("filename,width,height,class,xmin,ymin,xmax,ymax");
		
		File[] alltxts = indir.listFiles();
		
		int ctr=0;
		
		for(File f: alltxts) {
			String fn = f.getName();
			fn = fn.replaceAll("\\.txt", ".jpg");			
		
		List<String> flines = FileUtils.readLines(f);
		
		
		
		for(int i = 0 ; i< flines.size() ; ) {
			//read 2 lines 
			
		//	String classlabel = flines.get(i).trim();
			String everythingelse = flines.get(i+1).trim();
			
			BufferedImage bimg = ImageIO.read(new File(imgdir.getPath()+"/"+fn));
			int width          = bimg.getWidth();
			int height         = bimg.getHeight();
			
			String[] box = everythingelse.trim().split("\\s") ;//.replaceAll("\\s", ",");
			
			pwr.println(fn+","+width+","+height+","+"hand,"+(int)Double.parseDouble(box[0])+","+(int)Double.parseDouble(box[1])+","+(int)Double.parseDouble(box[2])+","+(int)Double.parseDouble(box[3]));
			i=i+2;
			
		}
		ctr++;
		
		}
		
		pwr.close();
		
	}

}
