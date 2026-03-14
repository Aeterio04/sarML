package com.hoh.SAR_Generator.model.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.UUID;

@Entity
@Table(name = "accounts")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Account {

    @Id
    private UUID id;

    @Column(name = "customer_id", nullable = false)
    private UUID customerId;

    @Column(name = "account_number", nullable = false)
    private String accountNumber;

    @Column(name = "account_type")
    private String accountType;

    @Column(name = "avg_monthly_bal")
    private BigDecimal avgMonthlyBal;

    @Column(name = "international_txn")
    private Boolean internationalTxn;

    @Column(name = "currency")
    private String currency;

    @Column(name = "open_date")
    private LocalDate openDate;

    @Column(name = "bank_id")
    private UUID bankId;

    @Column(name = "status")
    private String status;
}