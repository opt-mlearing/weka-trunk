package weka.classifiers.visualize;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;
import java.util.Random;

/**
 * visualize ros test sample.
 *
 * @author caogaoli
 * @date 2022/7/7 14:55
 */
@Slf4j
public class VisualizeTest {

    @Test
    public void testVisualROS() throws Exception {
        // 加载数据
        String file = "data/weather.nominal.arff";
        Instances data = ConverterUtils.DataSource.read(file);
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        // 评估分类器
        // 朴素贝叶斯
        NaiveBayes classifier = new NaiveBayes();
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1234L));
        // 第一步，生成可绘制的数据
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances cure = tc.getCurve(eval.predictions(), classIndex);
        // 第二步，将可绘制数据放入绘图容器
        PlotData2D plotData = new PlotData2D(cure);
        plotData.setPlotName(cure.relationName());
        plotData.addInstanceNumberAttribute();
        // 第三步，将可绘制数据放入绘图容器
        ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
        tvp.setROCString("(Area under ROC = " + Utils.doubleToString(ThresholdCurve.getROCArea(cure), 4) + ")");
        tvp.setName(cure.relationName());
        // 指定连接点
        boolean[] cp = new boolean[cure.numInstances()];
        Arrays.fill(cp, true);
        cp[0] = false;
        plotData.setConnectPoints(cp);
        // 添加绘图
        tvp.addPlot(plotData);
        // 第四步，将可视化面板添加到JFrame.
        final JFrame jFrame = new JFrame("WEKA ROS: " + tvp.getName());
        // 像素
        jFrame.setSize(500, 400);
        // 布局
        jFrame.getContentPane().setLayout(new BorderLayout());
        jFrame.getContentPane().add(tvp, BorderLayout.CENTER);
        // 自动隐藏释放窗体
        jFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        // 窗体可见性
        jFrame.setVisible(true);
    }


}
