/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    CostBenefitAnalysis
 *    Copyright (C) 2015 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.knowledgeflow.steps;

import weka.gui.knowledgeflow.KFGUIConsts;
import weka.gui.visualize.PlotData2D;
import weka.knowledgeflow.Data;
import weka.knowledgeflow.StepManager;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Step for storing and viewing threshold data in a cost-benefit visualization
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision: $
 */
@KFStep(name = "CostBenefitAnalysis", category = "Visualization",
        toolTipText = "View threshold data in an interactive cost-benefit visualization",
        iconPath = KFGUIConsts.BASE_ICON_PATH + "ModelPerformanceChart.gif")
public class CostBenefitAnalysis extends BaseSimpleDataVisualizer {

    private static final long serialVersionUID = 7756281775575854085L;

    @Override
    public List<String> getIncomingConnectionTypes() {
        return Arrays.asList(StepManager.CON_THRESHOLD_DATA);
    }

    /**
     * Process an incoming data payload (if the step accepts incoming connections)
     *
     * @param data the payload to process
     */
    @Override
    public void processIncoming(Data data) {
        getStepManager().processing();

        PlotData2D pd = data.getPrimaryPayload();

        getStepManager().logDetailed("Processing " + pd.getPlotName());
        m_data.add(data);
        getStepManager().finished();
    }

    /**
     * When running in a graphical execution environment a step can make one or
     * more popup Viewer components available. These might be used to display
     * results, graphics etc. Returning null indicates that the step has no such
     * additional graphical views. The map returned by this method should be keyed
     * by action name (e.g. "View results"), and values should be fully qualified
     * names of the corresponding StepInteractiveView implementation. Furthermore,
     * the contents of this map can (and should) be dependent on whether a
     * particular viewer should be made available - i.e. if execution hasn't
     * occurred yet, or if a particular incoming connection type is not present,
     * then it might not be possible to view certain results.
     * <p>
     * Viewers can implement StepInteractiveView directly (in which case they need
     * to extends JPanel), or extends the AbstractInteractiveViewer class. The
     * later extends JPanel, uses a BorderLayout, provides a "Close" button and a
     * method to add additional buttons.
     *
     * @return a map of viewer component names, or null if this step has no
     * graphical views
     */
    @Override
    public Map<String, String> getInteractiveViewers() {
        Map<String, String> views = new LinkedHashMap<String, String>();

        if (m_data.size() > 0) {
            views.put("Show plots",
                    "weka.gui.knowledgeflow.steps.CostBenefitAnalysisInteractiveView");
        }

        return views;
    }
}
