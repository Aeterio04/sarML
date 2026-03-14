package com.hoh.SAR_Generator.model.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "sar_narrative_versions")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class SarNarrativeVersion {

    @Id
    private UUID id;

    @Column(name = "case_id", nullable = false)
    private UUID caseId;

    @Column(name = "version_number", nullable = false)
    private Integer versionNumber;

    @Column(name = "content", columnDefinition = "TEXT")
    private String content;

    @Column(name = "generated_by")
    private String generatedBy;

    @Column(name = "model_name")
    private String modelName;

    @Column(name = "created_at")
    private Instant createdAt;

    @Column(name = "is_final")
    private Boolean isFinal;
}