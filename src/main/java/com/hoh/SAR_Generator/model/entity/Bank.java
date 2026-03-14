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
@Table(name = "bank")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Bank {

    @Id
    private UUID id;

    @Column(name = "bank_name", nullable = false)
    private String bankName;

    @Column(name = "swift_code", unique = true)
    private String swiftCode;

    @Column(name = "country")
    private String country;

    @Column(name = "risk_rating")
    private String riskRating;
}