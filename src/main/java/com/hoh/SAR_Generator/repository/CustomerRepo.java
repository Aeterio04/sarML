package com.hoh.SAR_Generator.repository;

import com.hoh.SAR_Generator.model.entity.Customer;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.UUID;

@Repository
public interface CustomerRepo extends JpaRepository<Customer, UUID> {
}