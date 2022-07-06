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
 * UpdateableClusterer.java
 * Copyright (C) 2006-2012 University of Waikato, Hamilton, New Zealand
 */

package weka.clusterers;

import weka.core.Instance;

/**
 * 增量聚类器.
 * 增量聚类器的训练一般分三个阶段完成：
 * 1）调用{@link Clusterer#buildClusterer(weka.core.Instances)}初始化模型；
 * 2）调用{@link UpdateableClusterer#updateClusterer(weka.core.Instance)}逐行更新模型；
 * 3）调用{@link UpdateableClusterer#updateFinished()}完成训练；
 * Interface to incremental cluster models that can learn using one instance at a time.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public interface UpdateableClusterer {

    /**
     * Adds an instance to the clusterer.
     *
     * @param newInstance the instance to be added
     * @throws Exception if something goes wrong
     */
    public void updateClusterer(Instance newInstance) throws Exception;

    /**
     * Signals the end of the updating.
     */
    public void updateFinished();
}
