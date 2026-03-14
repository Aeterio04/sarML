package com.hoh.SAR_Generator.model.dto;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CaseDTO {

    private UUID id;

    @NotNull(message = "Alert ID is required")
    private UUID alertId;

    @NotNull(message = "Assigned analyst ID is required")
    private UUID assignedTo;

    @NotBlank(message = "Status is required")
    private String status;

    @NotBlank(message = "Jurisdiction is required")
    private String jurisdiction;

    private Integer typologyId;

    private String priorityLevel;
}