package com.hoh.SAR_Generator.model.dto;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TransactionDTO {

    private UUID id;

    @NotNull(message = "Account ID is required")
    private UUID accountId;

    @NotNull(message = "Transaction date is required")
    private Instant txnDate;

    @NotBlank(message = "Transaction type is required")
    private String txnType;

    @NotNull(message = "Amount is required")
    private BigDecimal amount;

    private String currency;

    private String counterpartyName;

    private String counterpartyAccount;

    private String counterpartyCountry;

    private String channel;

    private Boolean isHighValue;

    private BigDecimal velocityScore;
}