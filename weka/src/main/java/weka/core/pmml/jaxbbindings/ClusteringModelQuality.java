//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.0-b52-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2013.12.20 at 12:48:21 PM GMT 
//


package weka.core.pmml.jaxbbindings;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ClusteringModelQuality element declaration.
 *
 * <p>The following schema fragment specifies the expected content contained within this class.
 *
 * <pre>
 * &lt;element name="ClusteringModelQuality">
 *   &lt;complexType>
 *     &lt;complexContent>
 *       &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *         &lt;attribute name="SSB" type="{http://www.dmg.org/PMML-4_1}NUMBER" />
 *         &lt;attribute name="SSE" type="{http://www.dmg.org/PMML-4_1}NUMBER" />
 *         &lt;attribute name="dataName" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;/restriction>
 *     &lt;/complexContent>
 *   &lt;/complexType>
 * &lt;/element>
 * </pre>
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "")
@XmlRootElement(name = "ClusteringModelQuality")
public class ClusteringModelQuality {

    @XmlAttribute(name = "SSB")
    protected Double ssb;
    @XmlAttribute(name = "SSE")
    protected Double sse;
    @XmlAttribute
    protected String dataName;

    /**
     * Gets the value of the ssb property.
     *
     * @return possible object is
     * {@link Double }
     */
    public Double getSSB() {
        return ssb;
    }

    /**
     * Sets the value of the ssb property.
     *
     * @param value allowed object is
     *              {@link Double }
     */
    public void setSSB(Double value) {
        this.ssb = value;
    }

    /**
     * Gets the value of the sse property.
     *
     * @return possible object is
     * {@link Double }
     */
    public Double getSSE() {
        return sse;
    }

    /**
     * Sets the value of the sse property.
     *
     * @param value allowed object is
     *              {@link Double }
     */
    public void setSSE(Double value) {
        this.sse = value;
    }

    /**
     * Gets the value of the dataName property.
     *
     * @return possible object is
     * {@link String }
     */
    public String getDataName() {
        return dataName;
    }

    /**
     * Sets the value of the dataName property.
     *
     * @param value allowed object is
     *              {@link String }
     */
    public void setDataName(String value) {
        this.dataName = value;
    }

}
