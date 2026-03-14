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
@Table(name = "cases")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Case {

    @Id
    private UUID id;

    @Column(name = "alert_id", nullable = false)
    private UUID alertId;

    @Column(name = "assigned_to", nullable = false)
    private UUID assignedTo;

    @Column(name = "status", nullable = false)
    private String status;

    @Column(name = "jurisdiction", nullable = false)
    private String jurisdiction;

    @Column(name = "typology_id")
    private Integer typologyId;

    @Column(name = "priority_level")
    private String priorityLevel;
}