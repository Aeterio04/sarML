package com.hoh.SAR_Generator.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SarResponseDTO {

    private UUID narrativeId;
    private UUID caseId;
    private Integer versionNumber;
    private String content;
    private String generatedBy;
    private String modelName;
    private Boolean isFinal;
}