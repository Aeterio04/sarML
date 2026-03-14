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
@Table(name = "reasoning_trace")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ReasoningTrace {

    @Id
    private UUID id;

    @Column(name = "case_id", nullable = false)
    private UUID caseId;

    @Column(name = "agent_name")
    private String agentName;

    @Column(name = "step_name")
    private String stepName;

    @Column(name = "input_reference")
    private String inputReference;

    @Column(name = "reasoning_output", columnDefinition = "TEXT")
    private String reasoningOutput;

    @Column(name = "created_at")
    private Instant createdAt;
}