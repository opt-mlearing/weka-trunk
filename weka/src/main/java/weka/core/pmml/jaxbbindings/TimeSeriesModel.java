//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.0-b52-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2013.12.20 at 12:48:21 PM GMT 
//


package weka.core.pmml.jaxbbindings;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElementRef;
import javax.xml.bind.annotation.XmlElementRefs;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TimeSeriesModel element declaration.
 *
 * <p>The following schema fragment specifies the expected content contained within this class.
 *
 * <pre>
 * &lt;element name="TimeSeriesModel">
 *   &lt;complexType>
 *     &lt;complexContent>
 *       &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *         &lt;sequence>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}Extension" maxOccurs="unbounded" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}MiningSchema"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}Output" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}ModelStats" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}ModelExplanation" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}LocalTransformations" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}TimeSeries" maxOccurs="3" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}SpectralAnalysis" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}ARIMA" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}ExponentialSmoothing" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}SeasonalTrendDecomposition" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}ModelVerification" minOccurs="0"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}Extension" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;/sequence>
 *         &lt;attribute name="algorithmName" type="{http://www.w3.org/2001/XMLSchema}string" />
 *         &lt;attribute name="bestFit" use="required" type="{http://www.dmg.org/PMML-4_1}TIMESERIES-ALGORITHM" />
 *         &lt;attribute name="functionName" use="required" type="{http://www.dmg.org/PMML-4_1}MINING-FUNCTION" />
 *         &lt;attribute name="isScorable" type="{http://www.w3.org/2001/XMLSchema}boolean" default="true" />
 *         &lt;attribute name="modelName" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;/restriction>
 *     &lt;/complexContent>
 *   &lt;/complexType>
 * &lt;/element>
 * </pre>
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
        "content"
})
@XmlRootElement(name = "TimeSeriesModel")
public class TimeSeriesModel {

    @XmlElementRefs({
            @XmlElementRef(name = "ModelExplanation", namespace = "http://www.dmg.org/PMML-4_1", type = ModelExplanation.class),
            @XmlElementRef(name = "ModelStats", namespace = "http://www.dmg.org/PMML-4_1", type = ModelStats.class),
            @XmlElementRef(name = "TimeSeries", namespace = "http://www.dmg.org/PMML-4_1", type = TimeSeries.class),
            @XmlElementRef(name = "Output", namespace = "http://www.dmg.org/PMML-4_1", type = Output.class),
            @XmlElementRef(name = "ExponentialSmoothing", namespace = "http://www.dmg.org/PMML-4_1", type = ExponentialSmoothing.class),
            @XmlElementRef(name = "LocalTransformations", namespace = "http://www.dmg.org/PMML-4_1", type = LocalTransformations.class),
            @XmlElementRef(name = "SeasonalTrendDecomposition", namespace = "http://www.dmg.org/PMML-4_1", type = JAXBElement.class),
            @XmlElementRef(name = "ARIMA", namespace = "http://www.dmg.org/PMML-4_1", type = JAXBElement.class),
            @XmlElementRef(name = "SpectralAnalysis", namespace = "http://www.dmg.org/PMML-4_1", type = JAXBElement.class),
            @XmlElementRef(name = "MiningSchema", namespace = "http://www.dmg.org/PMML-4_1", type = MiningSchema.class),
            @XmlElementRef(name = "ModelVerification", namespace = "http://www.dmg.org/PMML-4_1", type = ModelVerification.class),
            @XmlElementRef(name = "Extension", namespace = "http://www.dmg.org/PMML-4_1", type = Extension.class)
    })
    protected List<Object> content;
    @XmlAttribute
    protected String algorithmName;
    @XmlAttribute(required = true)
    protected TIMESERIESALGORITHM bestFit;
    @XmlAttribute(required = true)
    protected MININGFUNCTION functionName;
    @XmlAttribute
    protected Boolean isScorable;
    @XmlAttribute
    protected String modelName;

    /**
     * Gets the rest of the content model.
     *
     * <p>
     * You are getting this "catch-all" property because of the following reason:
     * The field name "Extension" is used by two different parts of a schema. See:
     * line 1156 of file:/home/david/workspace/weka/pmml-4-1.xsd
     * line 1144 of file:/home/david/workspace/weka/pmml-4-1.xsd
     * <p>
     * To get rid of this property, apply a property customization to one
     * of both of the following declarations to change their names:
     * Gets the value of the content property.
     *
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the content property.
     *
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getContent().add(newItem);
     * </pre>
     *
     *
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ModelExplanation }
     * {@link TimeSeries }
     * {@link ModelStats }
     * {@link Output }
     * {@link LocalTransformations }
     * {@link ExponentialSmoothing }
     * {@link JAXBElement }{@code <}{@link Object }{@code >}
     * {@link JAXBElement }{@code <}{@link Object }{@code >}
     * {@link JAXBElement }{@code <}{@link Object }{@code >}
     * {@link MiningSchema }
     * {@link ModelVerification }
     * {@link Extension }
     */
    public List<Object> getContent() {
        if (content == null) {
            content = new ArrayList<Object>();
        }
        return this.content;
    }

    /**
     * Gets the value of the algorithmName property.
     *
     * @return possible object is
     * {@link String }
     */
    public String getAlgorithmName() {
        return algorithmName;
    }

    /**
     * Sets the value of the algorithmName property.
     *
     * @param value allowed object is
     *              {@link String }
     */
    public void setAlgorithmName(String value) {
        this.algorithmName = value;
    }

    /**
     * Gets the value of the bestFit property.
     *
     * @return possible object is
     * {@link TIMESERIESALGORITHM }
     */
    public TIMESERIESALGORITHM getBestFit() {
        return bestFit;
    }

    /**
     * Sets the value of the bestFit property.
     *
     * @param value allowed object is
     *              {@link TIMESERIESALGORITHM }
     */
    public void setBestFit(TIMESERIESALGORITHM value) {
        this.bestFit = value;
    }

    /**
     * Gets the value of the functionName property.
     *
     * @return possible object is
     * {@link MININGFUNCTION }
     */
    public MININGFUNCTION getFunctionName() {
        return functionName;
    }

    /**
     * Sets the value of the functionName property.
     *
     * @param value allowed object is
     *              {@link MININGFUNCTION }
     */
    public void setFunctionName(MININGFUNCTION value) {
        this.functionName = value;
    }

    /**
     * Gets the value of the isScorable property.
     *
     * @return possible object is
     * {@link Boolean }
     */
    public boolean isIsScorable() {
        if (isScorable == null) {
            return true;
        } else {
            return isScorable;
        }
    }

    /**
     * Sets the value of the isScorable property.
     *
     * @param value allowed object is
     *              {@link Boolean }
     */
    public void setIsScorable(Boolean value) {
        this.isScorable = value;
    }

    /**
     * Gets the value of the modelName property.
     *
     * @return possible object is
     * {@link String }
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Sets the value of the modelName property.
     *
     * @param value allowed object is
     *              {@link String }
     */
    public void setModelName(String value) {
        this.modelName = value;
    }

}
