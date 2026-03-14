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
@Table(name = "transactions")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Transaction {

    @Id
    private UUID id;

    @Column(name = "account_id", nullable = false)
    private UUID accountId;

    @Column(name = "txn_date", nullable = false)
    private Instant txnDate;

    @Column(name = "txn_type", nullable = false)
    private String txnType;

    @Column(name = "amount", nullable = false)
    private BigDecimal amount;

    @Column(name = "currency")
    private String currency;

    @Column(name = "counterparty_name")
    private String counterpartyName;

    @Column(name = "counterparty_account")
    private String counterpartyAccount;

    @Column(name = "counterparty_country")
    private String counterpartyCountry;

    @Column(name = "channel")
    private String channel;

    @Column(name = "is_high_value")
    private Boolean isHighValue;

    @Column(name = "velocity_score")
    private BigDecimal velocityScore;
}