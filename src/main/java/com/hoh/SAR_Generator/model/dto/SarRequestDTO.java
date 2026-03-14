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
public class SarRequestDTO {

    @NotNull(message = "Case ID is required")
    private UUID caseId;

    @NotBlank(message = "Model name is required")
    private String modelName;

    @NotBlank(message = "Typology is required")
    private String typology;

    private String analystNotes;
}