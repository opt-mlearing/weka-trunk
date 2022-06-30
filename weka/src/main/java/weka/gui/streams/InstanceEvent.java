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
 *    InstanceEvent.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.streams;

import java.util.EventObject;

/**
 * An event encapsulating an instance stream event.
 *
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class InstanceEvent
        extends EventObject {

    /**
     * for serialization
     */
    private static final long serialVersionUID = 3207259868110667379L;

    /**
     * Specifies that the instance format is available
     */
    public static final int FORMAT_AVAILABLE = 1;

    /**
     * Specifies that an instance is available
     */
    public static final int INSTANCE_AVAILABLE = 2;

    /**
     * Specifies that the batch of instances is finished
     */
    public static final int BATCH_FINISHED = 3;

    private int m_ID;

    /**
     * Constructs an InstanceEvent with the specified source object and event
     * type
     *
     * @param source the object generating the InstanceEvent
     * @param ID     the type of the InstanceEvent
     */
    public InstanceEvent(Object source, int ID) {

        super(source);
        m_ID = ID;
    }

    /**
     * Get the event type
     *
     * @return the event type
     */
    public int getID() {

        return m_ID;
    }
}
