package com.hoh.SAR_Generator.model.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.UUID;

@Entity
@Table(name = "typology_patterns")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class TypologyPattern {

    @Id
    private UUID id;

    @Column(name = "typology_name", nullable = false)
    private String typologyName;

    @Column(name = "description")
    private String description;

    @Column(name = "trigger_rules", columnDefinition = "TEXT")
    private String triggerRules;

    @Column(name = "red_flags", columnDefinition = "TEXT")
    private String redFlags;

    @Column(name = "narrative_template", columnDefinition = "TEXT")
    private String narrativeTemplate;
}