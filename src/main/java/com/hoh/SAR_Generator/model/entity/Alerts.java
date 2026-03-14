package com.hoh.SAR_Generator.model.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "alerts")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Alerts {

    @Id
    private UUID id;

    @Column(name = "case_id")
    private UUID caseId;

    @Column(name = "alert_type")
    private String alertType;

    @Column(name = "rule_name")
    private String ruleName;

    @Column(name = "description")
    private String description;

    @Column(name = "triggered_at")
    private Instant triggeredAt;

    @Column(name = "risk_score")
    private BigDecimal riskScore;

    @Column(name = "confidence_score")
    private BigDecimal confidenceScore;
}